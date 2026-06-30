use clap::Parser;
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma, Rgb};
use std::io::{self, Write};
use std::str::FromStr;
use std::{
    ops::Sub,
    path::{Path, PathBuf},
    process,
};
use video_rs::frame::PixelFormat;
use video_rs::{Decoder, DecoderBuilder, Encoder, EncoderBuilder};

use crate::util::{image_to_frame, video_frame_to_image};
use crate::{
    algorithm::{CHARS, EDGE_CHARS},
    font_renderer::render_fonts_to_atlas,
};
use mimalloc::MiMalloc;

mod algorithm;
mod font_renderer;
#[macro_use]
mod util;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

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

        print!("{}", chars);
        image.save("./output/image.png")?;
    } else {
        let mut decoder = Decoder::new(args.input)?;
        let (src_w, src_h) = decoder.size();

        let out_w = ((src_w / cell_w) * cell_w) & !1;
        let out_h = ((src_h / cell_h) * cell_h) & !1;

        let mut encoder = Encoder::new(
            PathBuf::from_str("./output/render.mkv")?,
            video_rs::encode::Settings::preset_h264_custom(
                out_w as usize,
                out_h as usize,
                PixelFormat::YUV444P,
                video_rs::Options::preset_h264_realtime(),
            ),
        )?;

        let mut frames_processed = 0;
        for frame in decoder.decode_iter() {
            let (timestamp, frame) = match frame {
                Ok(f) => f,
                Err(_) => break,
            };

            let image = DynamicImage::ImageRgb8(video_frame_to_image(&frame));
            let (_, render) = algorithm::process_frame(&char_atlas, image, cell_w, cell_h, false)?;
            let render =
                image::imageops::crop_imm(&render.to_rgb8(), 0, 0, out_w, out_h).to_image();

            encoder.encode(&image_to_frame(&render), timestamp)?;
            frames_processed += 1;

            if frames_processed % 30 == 0 {
                print!("Frames processed: {}\r", frames_processed);
                io::stdout().flush()?;
            }
        }

        encoder.finish()?;
        println!("Done");
    }

    Ok(())
}
