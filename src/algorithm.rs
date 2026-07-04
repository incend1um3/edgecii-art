use crate::structure_tensor::{CellStructureTensors, StructureTensor};
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma, Rgb};
use linear_srgb::{default::linear_to_srgb, tf::srgb_to_linear};
use std::f32;

// dimmest to brightest
pub const CHARS: [char; 10] = [' ', '.', ':', '*', 'o', '?', '#', '%', '@', '■'];
// no order
pub const EDGE_CHARS: [char; 11] = ['|', '/', '-', '\\', '^', 'V', '<', '>', 'T', 'L', 'X'];

#[inline(always)]
fn angle_to_edge_index(theta: f32) -> usize {
    use f32::consts::PI;

    let t = (theta.rem_euclid(PI) + PI / 8.0) % PI; // shift by half a bin
    (t / PI * 4.0) as usize % 4
}

#[inline(always)]
fn angle_dist(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(f32::consts::PI);
    d.min(f32::consts::PI - d)
}

const EDGE_COHERANCE_THRESHOLD: f32 = 0.6;
const EDGE_ENERGY_THRESHOLD: f32 = 0.0025;
const SOBEL_MAX_RECIPROCAL_SQ: f32 = 1.0 / (1020.0 * 1020.0);

#[profiling::function]
fn try_special(tensors: &CellStructureTensors) -> Option<usize> {
    const ANGLE_TOLERANCE: f32 = f32::to_radians(30.0);
    const ALIGNMENT_TOLERANCE: f32 = 0.3;

    const THETA_FSLASH: f32 = f32::to_radians(45.0);
    const THETA_HORIZONTAL: f32 = 90.0f32.to_radians();
    const THETA_VERTICAL: f32 = 0.0;
    const THETA_BSLASH: f32 = f32::to_radians(45.0 + 90.0);

    let left = tensors.left();
    let right = tensors.right();
    let top = tensors.top();
    let bottom = tensors.bottom();

    let tl = &tensors.tl;
    let tr = &tensors.tr;
    let bl = &tensors.bl;
    let br = &tensors.br;

    let is_stroke = |tensor: &StructureTensor, angle: f32| {
        tensor.energy_avg() * SOBEL_MAX_RECIPROCAL_SQ > EDGE_ENERGY_THRESHOLD
            && tensor.coherence() > EDGE_COHERANCE_THRESHOLD * 0.8
            && angle_dist(tensor.theta(), angle) < ANGLE_TOLERANCE
    };

    let is_stroke_junction = |tensor: &StructureTensor, angle: f32| {
        tensor.energy_avg() * SOBEL_MAX_RECIPROCAL_SQ > EDGE_ENERGY_THRESHOLD
            && tensor.directional_alignment(angle) > ALIGNMENT_TOLERANCE
    };

    // V
    if is_stroke(&left, THETA_BSLASH) && is_stroke(&right, THETA_FSLASH) {
        return Some(5);
    }
    // ^
    if is_stroke(&left, THETA_FSLASH) && is_stroke(&right, THETA_BSLASH) {
        return Some(4);
    }
    // >
    if is_stroke(&top, THETA_BSLASH) && is_stroke(&bottom, THETA_FSLASH) {
        return Some(7);
    }
    // <
    if is_stroke(&top, THETA_FSLASH) && is_stroke(&bottom, THETA_BSLASH) {
        return Some(6);
    }
    // T
    if is_stroke_junction(&top, THETA_HORIZONTAL) && is_stroke(&bottom, THETA_VERTICAL) {
        return Some(8);
    }
    // L
    if tr.energy_avg() * SOBEL_MAX_RECIPROCAL_SQ < EDGE_ENERGY_THRESHOLD
        && is_stroke_junction(&left, THETA_VERTICAL)
        && is_stroke(&br, THETA_HORIZONTAL)
    {
        return Some(9);
    }
    // x
    if is_stroke(&tl, THETA_BSLASH)
        && is_stroke(&tr, THETA_FSLASH)
        && is_stroke(&bl, THETA_FSLASH)
        && is_stroke(&br, THETA_BSLASH)
    {
        return Some(10);
    }

    None
}

pub fn process_frame(
    char_atlas: &ndarray::Array2<u8>,
    image: DynamicImage,
    cell_w: u32,
    cell_h: u32,
    debug_output: bool,
) -> anyhow::Result<(Vec<usize>, DynamicImage)> {
    let image_luma8 = image.to_luma8();
    let temp = image.into_rgb8();
    let image_raw = temp.as_raw();

    let vertical_grad = {
        profiling::scope!("Vertical Sobel");
        imageproc::gradients::vertical_sobel(&image_luma8)
    };
    let horizontal_grad = {
        profiling::scope!("Horizontal Sobel");
        imageproc::gradients::horizontal_sobel(&image_luma8)
    };

    if debug_output {
        let angles = vertical_grad
            .pixels()
            .zip(horizontal_grad.pixels())
            .map(|(y, x)| f32::atan2(y[0] as f32, x[0] as f32))
            .collect::<Vec<f32>>();
        let angles_img = GrayImage::from_vec(
            image_luma8.width(),
            image_luma8.height(),
            angles
                .iter()
                .map(|theta| (((theta + f32::consts::PI) / (2.0 * f32::consts::PI)) * 255.0) as u8)
                .collect::<Vec<u8>>(),
        )
        .expect("Vector size matches width * height");

        angles_img.save("./.debug/angles.png").unwrap();

        let magnitudes = ImageBuffer::from_fn(vertical_grad.width(), vertical_grad.height(), |x, y| {
            let v = vertical_grad.get_pixel(x, y)[0] as f32;
            let h = horizontal_grad.get_pixel(x, y)[0] as f32;
            Luma([(v * v + h * h).sqrt().round().min(255.0) as u8])
        });
        magnitudes.save("./.debug/sobel.png")?;

        ImageBuffer::from_fn(vertical_grad.width(), vertical_grad.height(), |x, y| {
            let pixel_index = (y * image_luma8.width() + x) as usize;
            let theta = angles[pixel_index]; // -π to π
            let mag = magnitudes.get_pixel(x, y)[0] as f32 / 255.0;

            let hue = (theta + f32::consts::PI) / (2.0 * f32::consts::PI); // 0..1
            // simple HSV→RGB with S=1, V=mag
            let h = hue * 6.0;
            let i = h as u8;
            let f = h - i as f32;
            let (r, g, b) = match i % 6 {
                0 => (1.0, f, 0.0),
                1 => (1.0 - f, 1.0, 0.0),
                2 => (0.0, 1.0, f),
                3 => (0.0, 1.0 - f, 1.0),
                4 => (f, 0.0, 1.0),
                _ => (1.0, 0.0, 1.0 - f),
            };
            Rgb([
                (r * mag * 255.0) as u8,
                (g * mag * 255.0) as u8,
                (b * mag * 255.0) as u8,
            ])
        })
        .save("./.debug/edge_directions.png")?;
    }

    let img_width_snapped = (image_luma8.width() / cell_w as u32) * cell_w as u32;
    let img_height_snapped = (image_luma8.height() / cell_h as u32) * cell_h as u32;

    let srgb_to_linear_lut: [f32; 256] = std::array::from_fn(|i| srgb_to_linear(i as f32 / 255.0));
    let mut char_indices = Vec::with_capacity((img_width_snapped * img_height_snapped / (cell_w * cell_h)) as usize);

    let cell_quadrant_w = cell_w / 2;
    let cell_quadrant_h = cell_h / 2;

    for y in (0..(img_height_snapped)).step_by(cell_h as usize) {
        for x in (0..(img_width_snapped)).step_by(cell_w as usize) {
            let mut tensors = CellStructureTensors::new(cell_w, cell_h);
            let mut brightness_sum = 0.0f32;

            let mut loop_fn = |x1: u32, y1: u32, x2: u32, y2: u32, tensor: &mut StructureTensor| {
                let mut sum_gx_squared = 0;
                let mut sum_gxgy = 0;
                let mut sum_gy_squared = 0;

                for dy in y1..y2 {
                    for dx in x1..x2 {
                        brightness_sum = brightness_sum.algebraic_add(unsafe {
                            *srgb_to_linear_lut.get_unchecked(image_luma8.unsafe_get_pixel(x + dx, y + dy)[0] as usize)
                        });

                        let (gx, gy) = unsafe {
                            (
                                horizontal_grad.unsafe_get_pixel(x + dx, y + dy)[0],
                                vertical_grad.unsafe_get_pixel(x + dx, y + dy)[0],
                            )
                        };

                        sum_gx_squared += gx as i32 * gx as i32;
                        sum_gxgy += gx as i32 * gy as i32;
                        sum_gy_squared += gy as i32 * gy as i32;
                    }
                }

                tensor.gx_squared = sum_gx_squared as f32;
                tensor.gxgy = sum_gxgy as f32;
                tensor.gy_squared = sum_gy_squared as f32;
            };

            // Loop over each quadrant individually so there's no branching, allowing for autovectorization
            loop_fn(0, 0, cell_quadrant_w, cell_quadrant_h, &mut tensors.tl);
            loop_fn(
                cell_quadrant_w,
                0,
                cell_quadrant_w * 2,
                cell_quadrant_h,
                &mut tensors.tr,
            );
            loop_fn(
                0,
                cell_quadrant_h,
                cell_quadrant_w,
                cell_quadrant_h * 2,
                &mut tensors.bl,
            );
            loop_fn(
                cell_quadrant_w,
                cell_quadrant_h,
                cell_quadrant_w * 2,
                cell_quadrant_h * 2,
                &mut tensors.br,
            );

            let tensor_combined = tensors.combined();
            let special_edge_index = if tensor_combined.coherence() < EDGE_COHERANCE_THRESHOLD {
                try_special(&tensors)
            } else {
                None
            };

            let energy = tensor_combined.energy_avg() * SOBEL_MAX_RECIPROCAL_SQ;

            if let Some(i) = special_edge_index {
                char_indices.push(CHARS.len() + i);
            } else if energy > EDGE_ENERGY_THRESHOLD && tensor_combined.coherence() > EDGE_COHERANCE_THRESHOLD {
                char_indices.push(CHARS.len() + angle_to_edge_index(tensor_combined.theta()));
            } else {
                let brightness_avg = brightness_sum / (cell_w * cell_h) as f32;
                char_indices.push(((linear_to_srgb(brightness_avg)) * (CHARS.len() - 1) as f32).round() as usize);
            }
        }
    }

    let mut buffer = vec![0u8; (img_width_snapped * img_height_snapped * 3) as usize];
    {
        profiling::scope!("Rendering to Image");

        let cols = img_width_snapped / cell_w;

        for y in (0..img_height_snapped).step_by(cell_h as usize) {
            for x in (0..img_width_snapped).step_by(cell_w as usize) {
                let cell_x = x / cell_w;
                let cell_y = y / cell_h;
                let char_index = char_indices[(cell_y * cols + cell_x) as usize];
                let atlas_row = char_atlas.row(char_index);

                for dy in 0..cell_h {
                    for dx in 0..cell_w {
                        let pixel_index = (((y + dy) * img_width_snapped + x + dx) * 3) as usize;

                        let multiplier = atlas_row[(dy * cell_w + dx) as usize] as f32 / 255.0;

                        let r = unsafe { *image_raw.get_unchecked(pixel_index) };
                        let g = unsafe { *image_raw.get_unchecked(pixel_index + 1) };
                        let b = unsafe { *image_raw.get_unchecked(pixel_index + 2) };

                        buffer[pixel_index] =
                            unsafe { multiplier.algebraic_mul(r as f32).round().to_int_unchecked::<u8>() };
                        buffer[pixel_index + 1] =
                            unsafe { multiplier.algebraic_mul(g as f32).round().to_int_unchecked::<u8>() };
                        buffer[pixel_index + 2] =
                            unsafe { multiplier.algebraic_mul(b as f32).round().to_int_unchecked::<u8>() };
                    }
                }
            }
        }
    }

    let image = ImageBuffer::from_vec(img_width_snapped, img_height_snapped, buffer).unwrap();

    Ok((char_indices, DynamicImage::ImageRgb8(image)))
}

pub fn char_indices_to_string(
    cell_w: usize,
    cell_h: usize,
    img_width: usize,
    img_height: usize,
    indices: &[usize],
) -> String {
    let mut chars = String::with_capacity(img_height);

    for y in 0..(img_height / cell_h) {
        for x in 0..(img_width / cell_w) {
            let index = indices[(y * (img_width / cell_w) + x) as usize];

            let char = CHARS
                .iter()
                .chain(EDGE_CHARS.iter())
                .enumerate()
                .find(|(i, _)| *i == index)
                .map(|(_, c)| c)
                .unwrap();

            chars.push(*char);
        }

        chars.push('\n');
    }

    chars
}
