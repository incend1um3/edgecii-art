use clap::Parser;
use ffmpeg_sidecar::command::ffmpeg_is_installed;
use ffmpeg_sidecar::download::FfmpegDownloadProgressEvent;
use image::{DynamicImage, GrayImage, RgbImage};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use crate::ffmpeg_decoder::FfmpegVideoDecoder;
use crate::ffmpeg_encoder::FfmpegEncoder;
use crate::{
    algorithm::{CHARS, EDGE_CHARS},
    font_renderer::render_fonts_to_atlas,
};
use mimalloc::MiMalloc;

#[macro_use]
extern crate strum_macros;

mod algorithm;
mod ffmpeg_decoder;
mod ffmpeg_encoder;
mod font_renderer;
mod structure_tensor;
#[macro_use]
mod util;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Convert images and videos to ascii art with edge detection.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to image/video file
    #[arg(short, long)]
    input: PathBuf,

    /// Height of characters passed to FreeType (this may be different from the actual height of rendered cells)
    #[arg(long, value_parser = clap::value_parser!(u8).range(6..=100))]
    char_height: u8,

    /// Hardware accelerator to use for the encoder. Automatically defaults to a suitable one supported by the system.
    #[arg(long)]
    hw_accel: Option<ffmpeg_encoder::Vendor>,

    /// Compression level passed to the encoder.
    #[arg(long, default_value_t = ffmpeg_encoder::CompressionLevel::Balanced)]
    compression_level: ffmpeg_encoder::CompressionLevel,

    /// Quality preset passed to the encoder.
    #[arg(long, default_value_t = ffmpeg_encoder::Quality::High)]
    quality: ffmpeg_encoder::Quality,

    /// Video codec
    #[arg(short, long, default_value_t = ffmpeg_encoder::Codec::H265)]
    codec: ffmpeg_encoder::Codec,
}

static FRAMES_IN_QUEUE: AtomicU32 = AtomicU32::new(0);

enum DecoderThreadOutput {
    Data { id: u32, frame: RgbImage },
    End,
}

struct ProcessedFrame {
    id: u32,
    frame: RgbImage,
}

fn decode_thread(mut decoder: FfmpegVideoDecoder, mut tx: spmc::Sender<DecoderThreadOutput>) {
    let mut id = 0u32;
    loop {
        let frame = match decoder.next_frame() {
            Ok(Some(frame)) => frame,
            Ok(None) => break,
            Err(e) => {
                eprintln!("decode stopped early: {e}");
                break;
            }
        };

        while FRAMES_IN_QUEUE.load(Ordering::Relaxed) > 24 {
            std::thread::sleep(Duration::from_millis(400));
        }

        tx.send(DecoderThreadOutput::Data { id, frame }).unwrap();

        FRAMES_IN_QUEUE.fetch_add(1, Ordering::Relaxed);
        id += 1;
    }

    for _ in 0..(std::thread::available_parallelism().unwrap().get()) {
        let _ = tx.send(DecoderThreadOutput::End);
    }
}

fn encode_thread(mut encoder: FfmpegEncoder, rx: mpsc::Receiver<ProcessedFrame>) {
    let mut queue = HashMap::<u32, ProcessedFrame>::new();
    let mut next = 0u32;

    let mut timestamp = Instant::now();
    while let Ok(data) = rx.recv() {
        queue.insert(data.id, data);

        while let Some(data) = queue.remove(&next) {
            profiling::scope!("Encode");
            encoder.encode_frame(data.frame.into_raw()).unwrap();
            profiling::finish_frame!();

            FRAMES_IN_QUEUE.fetch_sub(1, Ordering::Relaxed);
            next += 1;

            if next % 30 == 0 {
                let fps = 30.0 / timestamp.elapsed().as_secs_f32();
                print!("\r\x1b[KProcessed {} frames, fps: {:.2}", next, fps);
                std::io::stdout().flush().unwrap();

                timestamp = Instant::now();
            }
        }
    }

    println!();
    encoder.finish().unwrap();
}

fn process_thread(
    char_atlas: Arc<ndarray::Array2<u8>>,
    cell_w: u32,
    cell_h: u32,
    out_w: u32,
    out_h: u32,
    rx: spmc::Receiver<DecoderThreadOutput>,
    tx: mpsc::Sender<ProcessedFrame>,
) {
    loop {
        let (id, frame) = {
            profiling::scope!("Wait for Decode");
            if let Ok(d) = rx.recv() {
                match d {
                    DecoderThreadOutput::Data { id, frame } => (id, frame),
                    DecoderThreadOutput::End => return,
                }
            } else {
                break;
            }
        };

        let (_, render) = {
            profiling::scope!("Process Frame");
            algorithm::process_frame(&char_atlas, DynamicImage::ImageRgb8(frame), cell_w, cell_h, false).unwrap()
        };

        let render = image::imageops::crop_imm(&render.into_rgb8(), 0, 0, out_w, out_h).to_image();

        tx.send(ProcessedFrame { id, frame: render }).unwrap();
    }
}

fn download_ffmpeg() {
    ffmpeg_sidecar::download::auto_download_with_progress(|p| {
        let message = match p {
            FfmpegDownloadProgressEvent::Starting => "Starting FFMPEG download...".into(),
            FfmpegDownloadProgressEvent::Downloading {
                total_bytes,
                downloaded_bytes,
            } => format!(
                "Downloading FFMPEG binaries: {:.1} kiB / {:.1} kiB",
                downloaded_bytes as f32 / 1024.0,
                total_bytes as f32 / 1024.0,
            ),
            FfmpegDownloadProgressEvent::UnpackingArchive => "Unpacking..".into(),
            FfmpegDownloadProgressEvent::Done => "Done!\n".into(),
        };

        print!("\r\x1b[K{}", message);
        let _ = std::io::stdout().flush();
    })
    .unwrap();
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if cfg!(debug_assertions) {
        let _ = std::fs::create_dir("./.debug");
    }
    let _ = std::fs::create_dir("./output");

    let is_video = match infer::get_from_path(&args.input)
        .map_err(|e| std::io::Error::new(e.kind(), format!("Failed to read input: {}", e)))?
        .map(|t| t.matcher_type())
    {
        Some(infer::MatcherType::Video) => true,
        Some(infer::MatcherType::Image) => false,
        _ => anyhow::bail!("Unsupported input file format"),
    };

    let input_file_stem = args
        .input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "render".to_string());

    let (char_atlas, cell_w, cell_h) = render_fonts_to_atlas(args.char_height as u32)?;

    if cfg!(debug_assertions) {
        let (data, _offset) = char_atlas.clone().into_raw_vec_and_offset();
        let atlas_img = GrayImage::from_raw(cell_w, (CHARS.len() + EDGE_CHARS.len()) as u32 * cell_h, data)
            .expect("buffer size matches dimensions");

        atlas_img.save("./.debug/atlas_vertical.png")?;
    }

    if !is_video {
        let image = image::open(&args.input)?;

        if image.width() < cell_w * 4 || image.height() < cell_h * 4 {
            anyhow::bail!(
                "Source media dimensions are too small relative to character size! Try lowering --char-height or using a larger image/video."
            );
        }

        let (char_indices, image) =
            algorithm::process_frame(&char_atlas, image, cell_w, cell_h, cfg!(debug_assertions))?;
        let chars = algorithm::char_indices_to_string(
            cell_w as usize,
            cell_h as usize,
            image.width() as usize,
            image.height() as usize,
            &char_indices,
        );

        std::fs::write(format!("./output/{input_file_stem}.txt"), chars)?;
        image.save(format!("./output/{input_file_stem}.png"))?;

        println!("Done. Output saved to ./output/ as {input_file_stem}.png and {input_file_stem}.txt");
    } else {
        download_ffmpeg();

        let (decode_tx, decode_rx) = spmc::channel();
        let (encode_tx, encode_rx) = mpsc::channel();

        let decoder = ffmpeg_decoder::create_decoder(&args.input)?;

        let (src_w, src_h) = decoder.size();
        let out_w = ((src_w / cell_w) * cell_w) & !1;
        let out_h = ((src_h / cell_h) * cell_h) & !1;

        if src_w < cell_w * 4 || src_h < cell_h * 4 {
            anyhow::bail!(
                "Source media dimensions are too small relative to character size! Try lowering --char-height or using a larger image/video."
            );
        }

        eprintln!("out_w={} out_h={} cell_w={} cell_h={}", out_w, out_h, cell_w, cell_h);

        let encoder = FfmpegEncoder::new(
            out_w,
            out_h,
            decoder.frame_rate_rational(),
            PathBuf::from(format!("./output/{input_file_stem}.mkv")),
            &args.input,
            args.codec,
            args.compression_level,
            ffmpeg_encoder::RateControl::Constant(args.quality),
            args.hw_accel,
        )?;

        let available_threads = std::thread::available_parallelism()?.get();
        let processing_threads = match (
            decoder.is_software,
            encoder.selected_vendor() == ffmpeg_encoder::Vendor::Software,
        ) {
            (true, true) => available_threads / 4,
            (true, _) | (_, true) => available_threads.saturating_sub(4),
            (false, false) => available_threads.saturating_sub(1),
        }
        .max(1);

        let decoder_handle = std::thread::spawn(move || decode_thread(decoder, decode_tx));
        let encoder_handle = std::thread::spawn(move || encode_thread(encoder, encode_rx));

        let char_atlas = Arc::new(char_atlas);
        let worker_handles: Vec<_> = (0..processing_threads)
            .map(|_| {
                let rx = decode_rx.clone();
                let tx = encode_tx.clone();
                let atlas = Arc::clone(&char_atlas);
                std::thread::spawn(move || process_thread(atlas, cell_w, cell_h, out_w, out_h, rx, tx))
            })
            .collect();

        decoder_handle.join().unwrap();
        worker_handles.into_iter().for_each(|h| h.join().unwrap());

        // Let the encoder thread exit
        drop(encode_tx);
        encoder_handle.join().unwrap();

        println!("Done. Output saved to ./output/{input_file_stem}.mkv");
    }

    Ok(())
}
