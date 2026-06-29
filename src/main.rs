use clap::Parser;
use image::{GenericImageView, GrayImage, ImageBuffer, Luma, Rgb};
use std::{
    ops::Sub,
    path::{Path, PathBuf},
    process,
};
use video_rs::{Decoder, Encoder};

use crate::{
    algorithm::{CHARS, EDGE_CHARS},
    font_renderer::render_fonts_to_atlas,
};

mod algorithm;
mod font_renderer;
#[macro_use]
mod util;

/// Convert images to ascii art with edge detection.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to image file
    #[arg(short, long)]
    input: PathBuf,

    /// Height of characters passed to FreeType (this may be different from the actual height of rendered cells)
    #[arg(short, long)]
    char_height: u8,
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

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if cfg!(debug_assertions) {
        let _ = std::fs::create_dir("./.debug");
    }
    let _ = std::fs::create_dir("./output");

    let file_bytes = std::fs::read(&args.input)?;
    let is_video = if infer::is_image(&file_bytes) {
        false
    } else if infer::is_video(&file_bytes) {
        true
    } else {
        println!("Unrecognized input file format");
        process::exit(-1);
    };

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

    if !is_video {
        let image = image::load_from_memory(&file_bytes)?;
        let (chars, image) =
            algorithm::process_frame(&char_atlas, image, cell_w, cell_h, cfg!(debug_assertions))?;

        print!("{}", String::from_iter(chars.iter()));
        image.save("./output/image.png")?;
    } else {
    }

    Ok(())
}
