use std::{f32, io::BufWriter};

use clap::{Arg, Parser};
use image::{ColorType, GenericImageView, GrayImage, ImageBuffer, Luma};

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
const CHARS: [char; 3] = ['.', ',', 'A'];
// no order
const EDGE_CHARS: [char; 6] = ['-', '/', '\\', '|', '_', '-'];

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
        let (data, _offset) = char_atlas.into_raw_vec_and_offset();
        let atlas_img = GrayImage::from_raw(
            cell_w as u32,
            (CHARS.len() as u32 * cell_h) as u32,
            data, // no copy
        )
        .expect("buffer size matches dimensions");

        atlas_img.save("./.debug/atlas_vertical.png")?;
    }

    let dog = difference_of_gaussians(&image_luma_f32, 1.5, 1.6 * 1.5);
    let dog_luma8 = luma_f32_to_u8(&dog);

    let image_luma8 = image.to_luma8();
    let (vertical_grad, horizontal_grad, magnitudes) = join3!(
        || imageproc::gradients::vertical_sobel(&dog_luma8),
        || imageproc::gradients::horizontal_sobel(&dog_luma8),
        || imageproc::gradients::sobel_gradients(&dog_luma8)
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
    }

    let img_width_snapped = (image_luma8.width() / cell_w as u32) * cell_w as u32;
    let img_height_snapped = (image_luma8.height() / cell_h as u32) * cell_h as u32;

    let mut chars = String::with_capacity(
        img_width_snapped as usize / cell_w as usize * img_height_snapped as usize
            / cell_h as usize,
    );
    for y in (0..(img_height_snapped)).step_by(cell_h as usize) {
        for x in (0..(img_width_snapped)).step_by(cell_w as usize) {
            const THRESHOLD: f32 = 10.0 / 360.0 * 2.0 * f32::consts::PI;

            for dy in 0..(cell_h as u32) {
                for dx in 0..(cell_w as u32) {
                    let pixel_index = ((y + dy) * image.width() + x + dx) as usize;
                    let theta = angles[pixel_index];

                    if dy == cell_h - 1 && dx == cell_w - 1 {
                        if (f32::consts::PI - theta).abs() < THRESHOLD || theta.abs() < THRESHOLD {
                            // edge pixel
                            chars.push('|');
                        } else {
                            chars.push('@');
                        }
                    }
                }
            }
        }
        chars.push('\n');
    }

    std::fs::write("ascii.txt", chars)?;

    // let max = gradients.pixels().map(|p| p[0]).max().unwrap();
    // let out = image::ImageBuffer::from_fn(gradients.width(), gradients.height(), |x, y| {
    //     let p = gradients.get_pixel(x, y)[0];
    //     image::Luma([((p as f32 / max as f32) * 255.0) as u8]) // scale u16 → u8
    // });
    // out.save("./out.png")?;

    Ok(())
}
