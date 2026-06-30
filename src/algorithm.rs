use crate::{join3, util::luma_f32_to_u8};
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma, Rgb};
use linear_srgb::{default::linear_to_srgb, tf::srgb_to_linear};
use rayon::iter::IntoParallelIterator;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use std::cell::LazyCell;
use std::sync::LazyLock;
use std::{f32, ffi::OsString, ops::Sub, process};

// dimmest to brightest
pub const CHARS: [char; 9] = [' ', '.', ':', '=', '+', '*', '#', '%', '@'];
// no order
pub const EDGE_CHARS: [char; 4] = ['|', '/', '_', '\\'];

pub fn angle_to_edge_index(theta: f32) -> usize {
    use f32::consts::PI;

    let t = (theta.rem_euclid(PI) + PI / 8.0) % PI; // shift by half a bin
    (t / PI * 4.0) as usize % 4
}

#[profiling::function]
pub fn difference_of_gaussians(
    img: &ImageBuffer<Luma<f32>, Vec<f32>>,
    sigma1: f32,
    sigma2: f32,
) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    assert!(sigma1 < sigma2);

    let (blur1, blur2) = {
        profiling::scope!("fast_blur");
        rayon::join(
            || image::imageops::fast_blur(img, sigma1),
            || image::imageops::fast_blur(img, sigma2),
        )
    };

    profiling::scope!("subtract");
    let raw = blur1
        .iter()
        .zip(blur2.iter())
        .map(|(a, b)| (a - b).abs())
        .collect::<Vec<_>>();

    ImageBuffer::from_raw(img.width(), img.height(), raw).unwrap()
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

    let (dog, vertical_grad, horizontal_grad) = {
        profiling::scope!("DoG + Sobel");
        join3!(
            || difference_of_gaussians(&image_luma_f32, 1.4, 1.6 * 1.4),
            || imageproc::gradients::vertical_sobel(&image_luma8),
            || imageproc::gradients::horizontal_sobel(&image_luma8)
        )
    };

    let dog_luma8 = {
        profiling::scope!("DoG cast + Sobel Magnitudes and Angles");
        luma_f32_to_u8(&dog)
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

        rayon::join(
            || angles_img.save("./.debug/angles.png").unwrap(),
            || dog_luma8.save("./.debug/dog.png").unwrap(),
        );

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

    // let mut chars = String::with_capacity(
    //     img_width_snapped as usize / cell_w as usize * img_height_snapped as usize
    //         / cell_h as usize,
    // );

    let char_indices = {
        profiling::scope!("Conversion Loop");
        (0..(img_height_snapped))
            .into_par_iter()
            .step_by(cell_h as usize)
            .map(|y| {
                let mut char_indices = Vec::with_capacity(cell_w as usize);

                for x in (0..(img_width_snapped)).step_by(cell_w as usize) {
                    const EDGE_PIXELS_THRESHOLD: f32 = 0.15;
                    let mut edge_histogram = [0u32; 4];
                    // let mut char_histogram = [0; CHARS.len()];
                    let mut edge_pixels = 0;
                    let mut brightness_sum = 0.0;

                    for dy in 0..(cell_h as u32) {
                        for dx in 0..(cell_w as u32) {
                            // let pixel_index = ((y + dy) * image.width() + x + dx) as usize;
                            let theta = unsafe {
                                libm::atan2f(
                                    vertical_grad.unsafe_get_pixel(x + dx, y + dy)[0] as f32,
                                    horizontal_grad.unsafe_get_pixel(x + dx, y + dy)[0] as f32,
                                )
                            };

                            brightness_sum += unsafe {
                                srgb_to_linear(image_luma_f32.unsafe_get_pixel(x + dx, y + dy)[0])
                            };

                            if unsafe { dog.unsafe_get_pixel(x + dx, y + dy)[0] } > 0.025 {
                                edge_histogram[angle_to_edge_index(theta)] += 1;
                                edge_pixels += 1;
                            }

                            // for char_index in 0..CHARS.len() {
                            //     let diff = (image_luma8.get_pixel(x + dx, y + dy)[0] as i16
                            //         - (char_atlas[[char_index, (dy * cell_w as u32 + dx) as usize]]) as i16)
                            //         .abs() as u32;
                            //     char_histogram[char_index] += diff;
                            // }
                        }
                    }

                    let edge_pixels_ratio = edge_pixels as f32 / (cell_w * cell_h) as f32;
                    if edge_pixels_ratio > EDGE_PIXELS_THRESHOLD {
                        char_indices.push(
                            CHARS.len()
                                + edge_histogram
                                    .iter()
                                    .enumerate()
                                    .max_by_key(|(_, samples)| *samples)
                                    .map(|(i, _)| i)
                                    .unwrap(),
                        );
                    } else {
                        let brightness_avg = brightness_sum / (cell_w * cell_h) as f32;
                        char_indices.push(
                            ((linear_to_srgb(brightness_avg)) * (CHARS.len() - 1) as f32).round()
                                as usize,
                        );
                    }
                }

                char_indices
            })
            .flatten()
            .collect::<Vec<usize>>()
    };

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
            let index = indices[(y * cell_h + x) as usize];

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
