use std::{f32, io::BufWriter, ops::Sub};

use clap::{Arg, Parser};
use image::{ColorType, GenericImageView, GrayImage, ImageBuffer, Luma, Rgb};

mod font_renderer;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    image: String,

    #[arg(short, long)]
    char_height: u8,
}

// dimmest to brightest
const CHARS: [char; 10] = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
// no order
const EDGE_CHARS: [char; 4] = ['|', '/', '_', '\\'];

fn angle_to_edge_index(mut theta: f32) -> usize {
    use f32::consts::PI;

    theta = theta % PI;
    if theta < 0.0 {
        theta += PI
    }
    (theta / PI * 4.0).min(3.0) as usize
}

fn compare_slices<T>(a: &[T], b: &[T]) -> T
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
    }

    let file_bytes = std::fs::read(args.image)?;
    let image = image::load_from_memory(&file_bytes)?;
    let image_luma_f32 = image.to_luma32f();

    let lib = freetype::Library::init()?;
    let face = lib.new_face("FiraSans-Regular.ttf", 0)?;
    face.set_pixel_sizes(0, args.char_height as u32)?;

    let size_metrics = face.size_metrics().unwrap();
    let ascender = (size_metrics.ascender >> 6) as i32;
    let cell_w = (size_metrics.max_advance >> 6) as u32;
    let cell_h = (size_metrics.height >> 6) as u32;

    // holds rendered font bitmaps
    let mut char_atlas: ndarray::Array2<u8> =
        ndarray::Array2::zeros((CHARS.len() + EDGE_CHARS.len(), (cell_h * cell_w) as usize));

    // https://freetype.org/freetype2/docs/tutorial/step2.html
    for (i, c) in CHARS.iter().chain(EDGE_CHARS.iter()).enumerate() {
        face.load_char(*c as usize, freetype::face::LoadFlag::RENDER)?;
        let glyph = face.glyph();
        let bitmap = glyph.bitmap();

        let buf = bitmap.buffer();
        let height = bitmap.rows() as i32;
        let width = bitmap.width() as i32;
        let left = glyph.bitmap_left() as i32;
        let top = glyph.bitmap_top() as i32;
        let advance = (glyph.advance().x >> 6) as i32;

        let mut j = 0;
        let y_offset = if ascender > top { ascender - top } else { 0 };
        let x_offset = (cell_w as i32 - advance) / 2;
        for y in 0..height {
            for x in 0..width {
                char_atlas[[
                    i,
                    ((y_offset + y) * cell_w as i32 + left + x_offset + x) as usize,
                ]] = buf[j];
                j += 1;
            }
        }
    }

    if cfg!(debug_assertions) {
        let (data, _offset) = char_atlas.clone().into_raw_vec_and_offset();
        let atlas_img =
            GrayImage::from_raw(cell_w as u32, (CHARS.len() as u32 * cell_h) as u32, data)
                .expect("buffer size matches dimensions");

        atlas_img.save("./.debug/atlas_vertical.png")?;
    }

    let dog = difference_of_gaussians(&image_luma_f32, 1.5, 1.6 * 1.5);
    let dog_luma8 = luma_f32_to_u8(&dog);

    let image_luma8 = image.to_luma8();
    let (vertical_grad, horizontal_grad, magnitudes) = join3!(
        || imageproc::gradients::vertical_sobel(&image_luma8),
        || imageproc::gradients::horizontal_sobel(&image_luma8),
        || imageproc::gradients::sobel_gradients(&image_luma8)
    );

    let angles = vertical_grad
        .pixels()
        .zip(horizontal_grad.pixels())
        .map(|(y, x)| libm::atan2f(y[0] as f32, x[0] as f32))
        .collect::<Vec<f32>>();

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

        angles_img.save("./.debug/angles.png")?;
        dog_luma8.save("./.debug/dog.png")?;

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

            for dy in 0..(cell_h as u32) {
                for dx in 0..(cell_w as u32) {
                    let pixel_index = ((y + dy) * image.width() + x + dx) as usize;
                    let sobel_magnitude = magnitudes.get_pixel(x + dx, y + dy)[0];
                    let theta = angles[pixel_index];
                    let luma = image_luma8.get_pixel(x + dx, y + dy)[0];

                    if dog.get_pixel(x + dx, y + dy)[0] > 0.02 {
                        edge_histogram[angle_to_edge_index(theta)] += 1;
                        edge_pixels += 1;
                    }

                    for char_index in 0..CHARS.len() {
                        let diff = (luma as i16
                            - (char_atlas[[char_index, (dy * cell_w as u32 + dx) as usize]]) as i16)
                            .abs() as u32;
                        char_histogram[char_index] += diff;
                    }
                }
            }

            let edge_pixels_ratio = edge_pixels as f32 / (cell_w * cell_h) as f32;
            if edge_pixels_ratio > 0.1 {
                chars.push(
                    EDGE_CHARS[edge_histogram
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, samples)| *samples)
                        .map(|(i, _)| i)
                        .unwrap()],
                );
            } else {
                chars.push(
                    CHARS[char_histogram
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, diff)| *diff)
                        .map(|(i, _)| i)
                        .unwrap()],
                )
            }
        }
        chars.push('\n');
    }

    std::fs::write("ascii.txt", &chars)?;

    let chars = chars.chars().filter(|&c| c != '\n').collect::<Vec<char>>();
    GrayImage::from_fn(img_width_snapped, img_height_snapped, |x, y| {
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

        Luma([char_atlas[[
            char_index as usize,
            (y % cell_h * cell_w + x % cell_w) as usize,
        ]]])
    })
    .save("render.png")?;

    // let max = gradients.pixels().map(|p| p[0]).max().unwrap();
    // let out = image::ImageBuffer::from_fn(gradients.width(), gradients.height(), |x, y| {
    //     let p = gradients.get_pixel(x, y)[0];
    //     image::Luma([((p as f32 / max as f32) * 255.0) as u8]) // scale u16 → u8
    // });
    // out.save("./out.png")?;

    Ok(())
}
