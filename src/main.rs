use clap::Parser;
use image::{GenericImageView, GrayImage, ImageBuffer, Luma, Rgb};
use linear_srgb::{default::linear_to_srgb, tf::srgb_to_linear};
use std::{f32, ffi::OsString, ops::Sub};

use crate::font_renderer::render_fonts_to_atlas;

mod font_renderer;

/// Convert images to ascii art with edge detection.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to image file
    #[arg(short, long)]
    image: OsString,

    /// Height of characters passed to FreeType (this may be different from the actual height of rendered cells)
    #[arg(short, long)]
    char_height: u8,
}

// dimmest to brightest
pub const CHARS: [char; 9] = [' ', '.', ':', '=', '+', '*', '#', '%', '@'];
// no order
pub const EDGE_CHARS: [char; 4] = ['|', '/', '_', '\\'];

fn angle_to_edge_index(theta: f32) -> usize {
    use f32::consts::PI;

    let t = (theta.rem_euclid(PI) + PI / 8.0) % PI; // shift by half a bin
    (t / PI * 4.0) as usize % 4
}

fn _compare_slices<T>(a: &[T], b: &[T]) -> T
where
    T: Copy + Sub<Output = T> + std::iter::Sum + num_traits::Signed,
{
    a.iter()
        .zip(b.iter())
        .map(|(pa, pb)| (*pb - *pa).abs())
        .sum()
}

macro_rules! join3 {
    ($a:expr, $b:expr, $c:expr) => {{
        let (a, (b, c)) = rayon::join($a, || rayon::join($b, $c));
        (a, b, c)
    }};
}

fn luma_f32_to_u8(img: &ImageBuffer<Luma<f32>, Vec<f32>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();

    ImageBuffer::from_fn(width, height, |x, y| {
        let Luma([v]) = *img.get_pixel(x, y);
        Luma([(v.clamp(0.0, 1.0) * 255.0).round() as u8])
    })
}

fn difference_of_gaussians(
    img: &ImageBuffer<Luma<f32>, Vec<f32>>,
    sigma1: f32,
    sigma2: f32,
) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    assert!(sigma1 < sigma2);

    let (blur1, blur2) = rayon::join(
        || imageproc::filter::gaussian_blur_f32(img, sigma1),
        || imageproc::filter::gaussian_blur_f32(img, sigma2),
    );

    let raw = blur1
        .iter()
        .zip(blur2.iter())
        .map(|(a, b)| (a - b).abs())
        .collect::<Vec<_>>();

    ImageBuffer::from_raw(img.width(), img.height(), raw).unwrap()
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if cfg!(debug_assertions) {
        let _ = std::fs::create_dir("./.debug");
        let _ = std::fs::create_dir("./output");
    }

    let file_bytes = std::fs::read(args.image)?;
    let image = image::load_from_memory(&file_bytes)?;
    let image_luma_f32 = image.to_luma32f();

    let (char_atlas, cell_w, cell_h) = render_fonts_to_atlas(args.char_height as u32)?;

    if cfg!(debug_assertions) {
        let (data, _offset) = char_atlas.clone().into_raw_vec_and_offset();
        let atlas_img = GrayImage::from_raw(
            cell_w as u32,
            ((CHARS.len() + EDGE_CHARS.len()) as u32 * cell_h) as u32,
            data,
        )
        .expect("buffer size matches dimensions");

        atlas_img.save("./.debug/atlas_vertical.png")?;
    }

    let image_luma8 = image.to_luma8();
    let (dog, vertical_grad, horizontal_grad) = join3!(
        || difference_of_gaussians(&image_luma_f32, 1.5, 1.6 * 1.5),
        || imageproc::gradients::vertical_sobel(&image_luma8),
        || imageproc::gradients::horizontal_sobel(&image_luma8)
    );
    let (dog_luma8, magnitudes, angles) = join3!(
        || luma_f32_to_u8(&dog),
        || {
            ImageBuffer::from_fn(vertical_grad.width(), vertical_grad.height(), |x, y| {
                let v = vertical_grad.get_pixel(x, y)[0] as f32;
                let h = horizontal_grad.get_pixel(x, y)[0] as f32;
                Luma([(v.powf(2.0) + h.powf(2.0)).sqrt().round() as u16])
            })
        },
        || vertical_grad
            .pixels()
            .zip(horizontal_grad.pixels())
            .map(|(y, x)| libm::atan2f(y[0] as f32, x[0] as f32))
            .collect::<Vec<f32>>()
    );

    if cfg!(debug_assertions) {
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

        ImageBuffer::from_fn(magnitudes.width(), magnitudes.height(), |x, y| {
            let Luma([v]) = *magnitudes.get_pixel(x, y);
            Luma([v.min(255) as u8])
        })
        .save("./.debug/sobel.png")?;

        ImageBuffer::from_fn(vertical_grad.width(), vertical_grad.height(), |x, y| {
            let pixel_index = (y * image.width() + x) as usize;
            let theta = angles[pixel_index]; // -π to π
            let mag = magnitudes.get_pixel(x, y)[0].min(1000) as f32 / 1000.0;

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

    let mut chars = String::with_capacity(
        img_width_snapped as usize / cell_w as usize * img_height_snapped as usize
            / cell_h as usize,
    );
    for y in (0..(img_height_snapped)).step_by(cell_h as usize) {
        for x in (0..(img_width_snapped)).step_by(cell_w as usize) {
            const EDGE_PIXELS_THRESHOLD: f32 = 0.1;
            let mut edge_histogram = [0u32; 4];
            let mut char_histogram = [0; CHARS.len()];
            let mut edge_pixels = 0;
            let mut brightness_sum = 0.0;

            for dy in 0..(cell_h as u32) {
                for dx in 0..(cell_w as u32) {
                    let pixel_index = ((y + dy) * image.width() + x + dx) as usize;
                    let theta = angles[pixel_index];

                    brightness_sum += srgb_to_linear(image_luma_f32.get_pixel(x + dx, y + dy)[0]);

                    if dog.get_pixel(x + dx, y + dy)[0] > 0.02 {
                        edge_histogram[angle_to_edge_index(theta)] += 1;
                        edge_pixels += 1;
                    }

                    for char_index in 0..CHARS.len() {
                        let diff = (image_luma8.get_pixel(x + dx, y + dy)[0] as i16
                            - (char_atlas[[char_index, (dy * cell_w as u32 + dx) as usize]]) as i16)
                            .abs() as u32;
                        char_histogram[char_index] += diff;
                    }
                }
            }

            let edge_pixels_ratio = edge_pixels as f32 / (cell_w * cell_h) as f32;
            if edge_pixels_ratio > EDGE_PIXELS_THRESHOLD {
                chars.push(
                    EDGE_CHARS[edge_histogram
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, samples)| *samples)
                        .map(|(i, _)| i)
                        .unwrap()],
                );
            } else {
                // chars.push(
                //     CHARS[char_histogram
                //         .iter()
                //         .enumerate()
                //         .min_by_key(|(_, diff)| *diff)
                //         .map(|(i, _)| i)
                //         .unwrap()],
                // )

                let brightness_avg = brightness_sum / (cell_w * cell_h) as f32 * 1.1;
                chars.push(
                    CHARS[((linear_to_srgb(brightness_avg)) * (CHARS.len() - 1) as f32).round()
                        as usize],
                );
            }
        }
        chars.push('\n');
    }

    std::fs::write("./output/ascii.txt", &chars)?;

    let chars = chars.chars().filter(|&c| c != '\n').collect::<Vec<char>>();
    ImageBuffer::from_fn(img_width_snapped, img_height_snapped, |x, y| {
        let cols = img_width_snapped / cell_w;
        let char_index = (y / cell_h) * cols + x / cell_w;
        let char = chars[char_index as usize];

        let char_index = CHARS
            .iter()
            .chain(EDGE_CHARS.iter())
            .enumerate()
            .find(|(_, c)| **c == char)
            .map(|(i, _)| i)
            .unwrap();

        let pixel = image.get_pixel(x, y);
        let text_multiplier = (char_atlas[[
            char_index as usize,
            (y % cell_h * cell_w + x % cell_w) as usize,
        ]] as f32
            / 255.0)
            .sqrt();

        Rgb([
            (pixel[0] as f32 / 255.0 * text_multiplier * 255.0).round() as u8,
            (pixel[1] as f32 / 255.0 * text_multiplier * 255.0).round() as u8,
            (pixel[2] as f32 / 255.0 * text_multiplier * 255.0).round() as u8,
        ])
    })
    .save("./output/render.png")?;

    Ok(())
}
