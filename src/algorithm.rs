use crate::{
    join3,
    structure_tensor::{CellStructureTensors, StructureTensor},
    util::luma_f32_to_u8,
};
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma, Rgb};
use linear_srgb::{default::linear_to_srgb, tf::srgb_to_linear};
use std::{f32, ffi::OsString, ops::Sub, process};

// dimmest to brightest
pub const CHARS: [char; 9] = [' ', '.', ':', '=', '+', '*', '#', '%', '@'];
// no order
pub const EDGE_CHARS: [char; 10] = ['|', '/', '-', '\\', '^', 'V', '<', '>', 'T', 'L'];

#[inline(always)]
pub fn angle_to_edge_index(theta: f32) -> usize {
    use f32::consts::PI;

    let t = (theta.rem_euclid(PI) + PI / 8.0) % PI; // shift by half a bin
    (t / PI * 4.0) as usize % 4
}

#[inline(always)]
pub fn angle_to_edge_char(theta: f32) -> char {
    EDGE_CHARS[angle_to_edge_index(theta)]
}

#[profiling::function]
pub fn process_frame(
    char_atlas: &ndarray::Array2<u8>,
    image: DynamicImage,
    cell_w: u32,
    cell_h: u32,
    debug_output: bool,
) -> anyhow::Result<(Vec<usize>, DynamicImage)> {
    let image_luma8 = image.to_luma8();
    let image_luma_f32 = image.to_luma32f();

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
            .map(|(y, x)| libm::atan2f(y[0] as f32, x[0] as f32))
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

        let magnitudes =
            ImageBuffer::from_fn(vertical_grad.width(), vertical_grad.height(), |x, y| {
                let v = vertical_grad.get_pixel(x, y)[0] as f32;
                let h = horizontal_grad.get_pixel(x, y)[0] as f32;
                Luma([(v * v + h * h).sqrt().round().min(255.0) as u8])
            });
        magnitudes.save("./.debug/sobel.png")?;

        ImageBuffer::from_fn(vertical_grad.width(), vertical_grad.height(), |x, y| {
            let pixel_index = (y * image.width() + x) as usize;
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

    let mut char_indices =
        Vec::with_capacity((img_width_snapped * img_height_snapped / (cell_w * cell_h)) as usize);
    for y in (0..(img_height_snapped)).step_by(cell_h as usize) {
        for x in (0..(img_width_snapped)).step_by(cell_w as usize) {
            const EDGE_COHERANCE_THRESHOLD: f32 = 0.6;
            const EDGE_ENERGY_THRESHOLD: f32 = 0.0025;

            let mut brightness_sum = 0.0;

            // structure tensor
            let mut tensors = CellStructureTensors::new(cell_w, cell_h);

            for dy in 0..(cell_h as u32) {
                for dx in 0..(cell_w as u32) {
                    // let pixel_index = ((y + dy) * image.width() + x + dx) as usize;
                    brightness_sum += unsafe {
                        srgb_to_linear(image_luma_f32.unsafe_get_pixel(x + dx, y + dy)[0])
                    };

                    let (gx, gy) = unsafe {
                        (
                            horizontal_grad.unsafe_get_pixel(x + dx, y + dy)[0] as f32,
                            vertical_grad.unsafe_get_pixel(x + dx, y + dy)[0] as f32,
                        )
                    };

                    tensors.accumulate(dx, dy, gx, gy);

                    // for char_index in 0..CHARS.len() {
                    //     let diff = (image_luma8.get_pixel(x + dx, y + dy)[0] as i16
                    //         - (char_atlas[[char_index, (dy * cell_w as u32 + dx) as usize]]) as i16)
                    //         .abs() as u32;
                    //     char_histogram[char_index] += diff;
                    // }
                }
            }

            const SOBEL_MAX_RECIPROCAL_SQ: f32 = 1.0 / (1020.0 * 1020.0);

            let test = |t: &StructureTensor| -> Option<char> {
                if t.energy_avg() * SOBEL_MAX_RECIPROCAL_SQ > EDGE_ENERGY_THRESHOLD
                    && t.coherence() > EDGE_COHERANCE_THRESHOLD * 0.8
                {
                    Some(angle_to_edge_char(t.theta()))
                } else {
                    None
                }
            };

            let top = test(&tensors.top);
            let bottom = test(&tensors.bottom);
            let left = test(&tensors.left);
            let right = test(&tensors.right);

            let special_edge_index = match (top, bottom, left, right) {
                (Some(t), Some(b), None, None) => match (t, b) {
                    ('-', '|') => Some(8),
                    ('\\', '/') => Some(7),
                    ('/', '\\') => Some(6),
                    (_, _) => None,
                },
                (None, None, Some(l), Some(r)) => match (l, r) {
                    ('\\', '/') => Some(5),
                    ('/', '\\') => Some(4),
                    (_, _) => None,
                },
                (Some(t), Some(b), Some(l), Some(r)) => match (t, b, l, r) {
                    ('|', '-', '|', '-') => Some(9),
                    (_, _, _, _) => None,
                },
                (_, _, _, _) => None,
            };

            let energy = tensors.combined.energy_avg() * SOBEL_MAX_RECIPROCAL_SQ;

            if let Some(i) = special_edge_index {
                char_indices.push(CHARS.len() + i);
            } else if energy > EDGE_ENERGY_THRESHOLD
                && tensors.combined.coherence() > EDGE_COHERANCE_THRESHOLD
            {
                char_indices.push(CHARS.len() + angle_to_edge_index(tensors.combined.theta()));
            } else {
                let brightness_avg = brightness_sum / (cell_w * cell_h) as f32;
                char_indices.push(
                    ((linear_to_srgb(brightness_avg)) * (CHARS.len() - 1) as f32).round() as usize,
                );
            }
        }
    }

    let mut buffer = Vec::with_capacity((img_width_snapped * img_height_snapped) as usize);
    {
        profiling::scope!("Rendering to Image");

        let lut: [f32; 256] = std::array::from_fn(|i| (i as f32 / 255.0).sqrt());
        for y in 0..img_height_snapped {
            for x in 0..img_width_snapped {
                let cols = img_width_snapped / cell_w;
                let index = (y / cell_h) * cols + x / cell_w;
                let char_index = char_indices[index as usize];

                let pixel = unsafe { image.unsafe_get_pixel(x, y) };
                let text_multiplier = lut[char_atlas[[
                    char_index as usize,
                    (y % cell_h * cell_w + x % cell_w) as usize,
                ]] as usize];

                buffer.push((pixel[0] as f32 / 255.0 * text_multiplier * 255.0).round() as u8);
                buffer.push((pixel[1] as f32 / 255.0 * text_multiplier * 255.0).round() as u8);
                buffer.push((pixel[2] as f32 / 255.0 * text_multiplier * 255.0).round() as u8);
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
