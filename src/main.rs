#![feature(float_algebraic)]

use clap::Parser;
use ffmpeg_sidecar::download::FfmpegDownloadProgressEvent;
use image::{DynamicImage, GrayImage};
use std::collections::HashMap;
use std::io::Write;
use std::str::FromStr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};
use std::{
    ops::Sub,
    path::{Path, PathBuf},
    process,
};
use video_rs::{Decoder, DecoderBuilder};

use crate::ffmpeg_encoder::FfmpegEncoder;
use crate::util::{image_to_frame, video_frame_to_image};
use crate::{
    algorithm::{CHARS, EDGE_CHARS},
    font_renderer::render_fonts_to_atlas,
};
use mimalloc::MiMalloc;

#[macro_use]
extern crate strum_macros;

mod algorithm;
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
    #[arg(short, long)]
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

fn _compare_slices<T>(a: &[T], b: &[T]) -> T
where
    T: Copy + Sub<Output = T> + std::iter::Sum + num_traits::Signed,
{
    a.iter().zip(b.iter()).map(|(pa, pb)| (*pb - *pa).abs()).sum()
}

static FRAMES_IN_QUEUE: AtomicU32 = AtomicU32::new(0);

enum DecoderThreadOutput {
    Data {
        id: u32,
        timestamp: video_rs::Time,
        frame: ndarray::Array3<u8>,
    },
    End,
}

struct ProcessedFrame {
    id: u32,
    frame: ndarray::Array3<u8>,
}

fn decode_thread(mut decoder: Decoder, mut tx: spmc::Sender<DecoderThreadOutput>) {
    let mut id = 0u32;
    for frame in decoder.decode_iter() {
        let (timestamp, frame) = match frame {
            Ok(f) => f,
            Err(_) => break,
        };

        while FRAMES_IN_QUEUE.load(Ordering::Relaxed) > 24 {
            std::thread::sleep(Duration::from_millis(400));
        }

        tx.send(DecoderThreadOutput::Data { id, timestamp, frame }).unwrap();

        FRAMES_IN_QUEUE.fetch_add(1, Ordering::Relaxed);
        id += 1;
    }

    tx.send(DecoderThreadOutput::End).unwrap();
}

fn encode_thread(mut encoder: FfmpegEncoder, rx: mpsc::Receiver<ProcessedFrame>) {
    let mut queue = HashMap::<u32, ProcessedFrame>::new();
    let mut next = 0u32;

    let mut timestamp = Instant::now();
    while let Ok(data) = rx.recv() {
        queue.insert(data.id, data);

        while let Some(data) = queue.remove(&next) {
            profiling::scope!("Encode");
            encoder.encode_frame(data.frame).unwrap();
            profiling::finish_frame!();

            FRAMES_IN_QUEUE.fetch_sub(1, Ordering::Relaxed);
            next += 1;

            if next % 30 == 0 {
                let fps = 30.0 / timestamp.elapsed().as_secs_f32();
                print!("Processed {} frames, fps: {:.2}\t\r", next, fps);
                std::io::stdout().flush().unwrap();

                timestamp = Instant::now();
            }
        }
    }

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
        let (id, _timestamp, frame) = {
            profiling::scope!("Wait for Decode");
            if let Ok(d) = rx.recv() {
                match d {
                    DecoderThreadOutput::Data { id, timestamp, frame } => (id, timestamp, frame),
                    DecoderThreadOutput::End => return,
                }
            } else {
                break;
            }
        };

        let image = DynamicImage::ImageRgb8(video_frame_to_image(frame));

        let (_, render) = {
            profiling::scope!("Process Frame");
            algorithm::process_frame(&char_atlas, image, cell_w, cell_h, false).unwrap()
        };

        let render = image::imageops::crop_imm(&render.to_rgb8(), 0, 0, out_w, out_h).to_image();

        tx.send(ProcessedFrame {
            id,
            frame: image_to_frame(&render),
        })
        .unwrap();
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

        print!("{}\t\t\t\r", message);
        let _ = std::io::stdout().flush();
    })
    .unwrap();

    println!();
}

fn create_decoder(file: &Path) -> anyhow::Result<video_rs::Decoder> {
    if !file.exists() || !file.is_file() {
        anyhow::bail!("Input file not found: {}", file.display());
    }

    let accelerators_available = video_rs::hwaccel::HardwareAccelerationDeviceType::list_available();

    use video_rs::hwaccel::HardwareAccelerationDeviceType::*;
    let accel_order = [Cuda, D3D11Va, Dxva2, Vdpau, VaApi, Qsv, VideoToolbox];
    let accelerators_available = accel_order.iter().filter(|a| accelerators_available.contains(a));

    video_rs::ffmpeg::log::set_level(video_rs::ffmpeg::log::Level::Quiet);
    for accel in accelerators_available {
        let decoder = DecoderBuilder::new(file).with_hardware_acceleration(*accel).build();

        if let Ok(mut d) = decoder
            && d.decode_raw().is_ok()
        {
            d.seek_to_start()?;
            video_rs::ffmpeg::log::set_level(video_rs::ffmpeg::log::Level::Warning);

            println!("Using hardware accelerated decoding through {:#?}", accel);
            return Ok(d);
        }
    }

    println!(
        "Warning: using software decoding because no working hardware decoder implementation was found. This will be EXTREMELY slow!"
    );
    video_rs::ffmpeg::log::set_level(video_rs::ffmpeg::log::Level::Warning);
    Ok(Decoder::new(file)?)
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

        let (char_indices, image) =
            algorithm::process_frame(&char_atlas, image, cell_w, cell_h, cfg!(debug_assertions))?;
        let chars = algorithm::char_indices_to_string(
            cell_w as usize,
            cell_h as usize,
            image.width() as usize,
            image.height() as usize,
            &char_indices,
        );

        print!("{}", chars);
        std::fs::write("./output/render.txt", chars)?;
        image.save("./output/render.png")?;
    } else {
        download_ffmpeg();

        let (decode_tx, decode_rx) = spmc::channel();
        let (encode_tx, encode_rx) = mpsc::channel();

        let decoder = create_decoder(&args.input)?;

        let (src_w, src_h) = decoder.size();
        let out_w = ((src_w / cell_w) * cell_w) & !1;
        let out_h = ((src_h / cell_h) * cell_h) & !1;
        eprintln!("out_w={} out_h={} cell_w={} cell_h={}", out_w, out_h, cell_w, cell_h);

        let encoder = FfmpegEncoder::new(
            out_w,
            out_h,
            decoder.frame_rate(),
            PathBuf::from_str("./output/render.mkv")?,
            &args.input,
            args.codec,
            args.compression_level,
            ffmpeg_encoder::RateControl::Constant(args.quality),
            args.hw_accel,
        )?;

        let decoder_handle = std::thread::spawn(move || decode_thread(decoder, decode_tx));
        let encoder_handle = std::thread::spawn(move || encode_thread(encoder, encode_rx));

        let char_atlas = Arc::new(char_atlas);
        let worker_handles: Vec<_> = (0..std::thread::available_parallelism().unwrap().get() - 2)
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

        println!("Done");
    }

    Ok(())
}
