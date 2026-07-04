use std::{
    io::Write,
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    time::{Duration, Instant},
};

use ffmpeg_sidecar::paths::ffmpeg_path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, IntoStaticStr, clap::ValueEnum)]
pub enum Vendor {
    Nvenc,
    Amf,
    Vaapi,
    Qsv,
    VideoToolbox,
    Software,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, IntoStaticStr)]
pub enum Codec {
    H264,
    H265,
    Av1,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum, strum_macros::Display)]
pub enum Quality {
    #[strum(serialize = "low")]
    Low,
    #[strum(serialize = "balanced")]
    Balanced,
    #[strum(serialize = "high")]
    High,
    #[strum(serialize = "perceptually_lossless")]
    PerceptuallyLossless,
}

impl Quality {
    /// Constant-quality value on the H.264/H.265 QP / CRF scale (0..=51, LOWER = better).
    /// Also used for QSV constant quality: `-global_quality` (ICQ) stays on this ~1..=51
    /// scale for AV1 too (AV1-QSV guides use values like 24 for high quality), so QSV does
    /// NOT need the wider hardware-AV1 scale below.
    fn h26x_qp(self) -> u32 {
        match self {
            Self::Low => 32,
            Self::Balanced => 23,
            Self::High => 19,
            Self::PerceptuallyLossless => 14,
        }
    }

    /// Constant-quality CRF for libsvtav1 (0..=63, LOWER = better).
    fn av1_crf(self) -> u32 {
        match self {
            Self::Low => 45,
            Self::Balanced => 30,
            Self::High => 22,
            Self::PerceptuallyLossless => 16,
        }
    }

    /// Constant-QP for the "raw quantizer index" hardware AV1 encoders — `av1_nvenc`,
    /// `av1_amf`, `av1_vaapi` — whose `-qp`/`-qp_i` run on AV1's 0..=255 q_index scale
    /// (LOWER = better), NOT H.264/H.265's 0..=51. AOM/SVT map a 0..=63 quantizer onto
    /// 0..=255 by ×4, so we reuse the libsvtav1 CRF value as that 0..=63 quantizer.
    /// Note: fixed-function AV1 is less efficient than SVT-AV1 at equal quantizer, so at a
    /// given `Quality` these will look a touch softer / larger than the software path.
    /// Does NOT apply to `av1_qsv` (ICQ, see `h26x_qp`) or VideoToolbox (no HW AV1 encoder).
    fn av1_hw_qp(self) -> u32 {
        self.av1_crf() * 4
    }

    /// VideoToolbox `-q:v` quality (1..=100, HIGHER = better — inverted vs. the others).
    fn videotoolbox_qv(self) -> u32 {
        match self {
            Self::Low => 35,
            Self::Balanced => 50,
            Self::High => 65,
            Self::PerceptuallyLossless => 85,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum RateControl {
    /// CRF / CQP / ICQ
    Constant(Quality),
    Bitrate {
        avg_kbps: u32,
        max_kbps: Option<u32>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum, strum_macros::Display)]
pub enum CompressionLevel {
    #[strum(serialize = "fast")]
    Fast,
    #[strum(serialize = "balanced")]
    Balanced,
    #[strum(serialize = "high")]
    High,
}

impl CompressionLevel {
    fn idx(self) -> usize {
        match self {
            Self::Fast => 0,
            Self::Balanced => 1,
            Self::High => 2,
        }
    }
}

pub struct FfmpegEncoder {
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    output: PathBuf,
    width: u32,
    height: u32,
    vendor: Vendor,
    encoder: String,
}

impl FfmpegEncoder {
    pub fn new(
        width: u32,
        height: u32,
        fps: f32,
        output: PathBuf,
        codec: Codec,
        quality: CompressionLevel,
        rate: RateControl,
        preferred_vendor: Option<Vendor>,
    ) -> anyhow::Result<Self> {
        let (vendor, encoder) = select_encoder(codec, preferred_vendor)
            .ok_or_else(|| anyhow::anyhow!("no working encoder found for {codec:?}"))?;

        if let Some(v) = preferred_vendor
            && v != vendor
        {
            println!("Failed to select {:?} for encoding", v);
        }

        println!("selected encoder: {encoder} (vendor: {vendor:?})");

        let (pre_input, pix_or_filter) = hw_setup_args(vendor);
        let mut args: Vec<String> = Vec::new();

        args.extend(["-hide_banner", "-v", "error", "-nostdin"].map(String::from));
        // Global / pre-input hw device setup (e.g. -vaapi_device ...).
        args.extend(pre_input);
        // Raw RGB input over the pipe.
        args.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "-s"].map(String::from));
        args.push(format!("{width}x{height}"));
        args.push("-r".into());
        args.push(format!("{fps}"));
        args.extend(["-i", "pipe:0"].map(String::from));
        // Output pixel format / upload filter for the chosen backend.
        args.extend(pix_or_filter);
        // The encoder itself.
        args.push("-c:v".into());
        args.push(encoder.clone());
        // Rate control + preset.
        args.extend(create_rate_and_preset_args(codec, vendor, quality, &rate));
        // Overwrite output.
        args.push("-y".into());
        args.push(output.to_string_lossy().into_owned());

        let mut child = Command::new(ffmpeg_path())
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit()) // -v error keeps this quiet unless something breaks
            .spawn()?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("failed to capture ffmpeg stdin"))?;

        Ok(Self {
            child: Some(child),
            stdin: Some(stdin),
            output,
            width,
            height,
            vendor,
            encoder,
        })
    }

    pub fn encode_frame(&mut self, frame: ndarray::Array3<u8>) -> anyhow::Result<()> {
        let expected = (self.width * self.height * 3) as usize;
        let (rgb, _) = frame
            .as_standard_layout()
            .to_owned()
            .into_raw_vec_and_offset();

        anyhow::ensure!(
            rgb.len() == expected,
            "frame size mismatch: got {} bytes, expected {} ({}x{}x3)",
            rgb.len(),
            expected,
            self.width,
            self.height
        );

        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("encoder already finished"))?;
        stdin.write_all(&rgb)?;
        Ok(())
    }

    /// Flush remaining frames and wait for ffmpeg to finalize the file.
    pub fn finish(mut self) -> anyhow::Result<()> {
        self.finish_inner()
    }

    fn finish_inner(&mut self) -> anyhow::Result<()> {
        // Dropping stdin sends EOF so ffmpeg flushes and writes the trailer.
        drop(self.stdin.take());
        if let Some(mut child) = self.child.take() {
            let status = child.wait()?;
            anyhow::ensure!(
                status.success(),
                "ffmpeg exited with status {status} while writing {}",
                self.output.display()
            );
        }
        Ok(())
    }

    pub fn selected_encoder(&self) -> &str {
        &self.encoder
    }

    pub fn selected_vendor(&self) -> Vendor {
        self.vendor
    }
}

impl Drop for FfmpegEncoder {
    fn drop(&mut self) {
        let _ = self.finish_inner();
    }
}

fn encoder_name(codec: Codec, vendor: Vendor) -> String {
    match vendor {
        Vendor::Software => match codec {
            Codec::H264 => "libx264".into(),
            Codec::H265 => "libx265".into(),
            Codec::Av1 => "libsvtav1".into(),
        },
        _ => {
            let codec_tok = match codec {
                Codec::H264 => "h264",
                Codec::H265 => "hevc",
                Codec::Av1 => "av1",
            };
            let vendor_str: &'static str = vendor.into();
            format!("{codec_tok}_{}", vendor_str.to_lowercase())
        }
    }
}

fn vendor_order() -> &'static [Vendor] {
    if cfg!(target_os = "macos") {
        &[Vendor::VideoToolbox, Vendor::Software]
    } else if cfg!(target_os = "windows") {
        &[Vendor::Nvenc, Vendor::Amf, Vendor::Qsv, Vendor::Software]
    } else {
        // linux / unix-like
        &[
            Vendor::Nvenc,
            Vendor::Vaapi,
            Vendor::Amf,
            Vendor::Qsv,
            Vendor::Software,
        ]
    }
}

/// Extra args needed to get RGB frames into the encoder for each backend.
/// Returns (pre-input global args, output-side pixel-format / upload args).
///
/// The VAAPI device path and QSV device init below are best-effort defaults;
/// on unusual setups you may need to adjust `/dev/dri/renderD128`.
fn hw_setup_args(vendor: Vendor) -> (Vec<String>, Vec<String>) {
    match vendor {
        // These uploaders take software P010 (10-bit) frames and upload internally.
        Vendor::Nvenc | Vendor::Amf | Vendor::VideoToolbox => {
            (vec![], vec!["-pix_fmt".into(), "p010le".into()])
        }
        Vendor::Vaapi => (
            vec!["-vaapi_device".into(), "/dev/dri/renderD128".into()],
            vec!["-vf".into(), "format=p010le,hwupload".into()],
        ),
        Vendor::Qsv => (
            vec![
                "-init_hw_device".into(),
                "qsv=hw".into(),
                "-filter_hw_device".into(),
                "hw".into(),
            ],
            vec![
                "-vf".into(),
                "format=p010le,hwupload=extra_hw_frames=64".into(),
            ],
        ),
        Vendor::Software => (vec![], vec!["-pix_fmt".into(), "yuv420p10le".into()]),
    }
}

fn run_probe(codec: Codec, vendor: Vendor) -> bool {
    let name = encoder_name(codec, vendor);
    let (pre_input, pix_or_filter) = hw_setup_args(vendor);

    let mut ffmpeg = Command::new(ffmpeg_path());
    ffmpeg.args(["-hide_banner", "-v", "error", "-nostdin"]);
    ffmpeg.args(&pre_input);
    ffmpeg.args(["-f", "lavfi", "-i", "color=c=black:s=128x128:r=30:d=0.1"]);
    ffmpeg.args(&pix_or_filter);
    ffmpeg.args(["-c:v", &name]);
    ffmpeg.args(["-frames:v", "1", "-f", "null", "-"]);
    ffmpeg
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    run_with_timeout(ffmpeg, Duration::from_secs(8))
}

fn run_with_timeout(mut cmd: Command, timeout: Duration) -> bool {
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(_) => return false,
    };
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(s)) => return s.success(),
            Ok(None) if start.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return false;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(_) => return false,
        }
    }
}

fn create_rate_and_preset_args(
    codec: Codec,
    vendor: Vendor,
    compression_level: CompressionLevel,
    rate: &RateControl,
) -> Vec<String> {
    let mut a: Vec<String> = Vec::new();
    let mut push = |args: &[&str]| a.extend(args.iter().map(|s| s.to_string()));
    let q = compression_level.idx();

    // Constant-QP value for the raw-quantizer hardware encoders (nvenc/amf/vaapi): AV1
    // rides the 0..=255 q_index scale while H.264/H.265 use 0..=51. QSV (ICQ) and
    // VideoToolbox keep their own scales and don't call this.
    let hw_const_qp = |quality: Quality| -> u32 {
        if codec == Codec::Av1 {
            quality.av1_hw_qp()
        } else {
            quality.h26x_qp()
        }
    };

    match vendor {
        Vendor::Nvenc => {
            push(&["-preset", ["p1", "p4", "p7"][q]]);
            match rate {
                RateControl::Constant(quality) => {
                    push(&["-rc", "constqp", "-qp"]);
                    a.push(hw_const_qp(*quality).to_string());
                }
                RateControl::Bitrate { avg_kbps, max_kbps } => match max_kbps {
                    Some(m) => {
                        push(&["-rc", "vbr"]);
                        a.push("-b:v".into());
                        a.push(format!("{avg_kbps}k"));
                        a.push("-maxrate".into());
                        a.push(format!("{m}k"));
                        a.push("-bufsize".into());
                        a.push(format!("{}k", m * 2));
                    }
                    None => {
                        push(&["-rc", "cbr"]);
                        a.push("-b:v".into());
                        a.push(format!("{avg_kbps}k"));
                    }
                },
            }
        }
        Vendor::Amf => {
            push(&["-quality", ["speed", "balanced", "quality"][q]]);
            match rate {
                RateControl::Constant(quality) => {
                    let qp = hw_const_qp(*quality).to_string();
                    a.push("-rc".into());
                    a.push("cqp".into());
                    a.push("-qp_i".into());
                    a.push(qp.clone());
                    a.push("-qp_p".into());
                    a.push(qp.clone());
                    a.push("-qp_b".into());
                    a.push(qp);
                }
                RateControl::Bitrate { avg_kbps, max_kbps } => match max_kbps {
                    Some(m) => {
                        push(&["-rc", "vbr_peak"]);
                        a.push("-b:v".into());
                        a.push(format!("{avg_kbps}k"));
                        a.push("-maxrate".into());
                        a.push(format!("{m}k"));
                    }
                    None => {
                        push(&["-rc", "cbr"]);
                        a.push("-b:v".into());
                        a.push(format!("{avg_kbps}k"));
                    }
                },
            }
        }
        Vendor::Qsv => {
            push(&["-preset", ["veryfast", "medium", "veryslow"][q]]);
            match rate {
                RateControl::Constant(quality) => {
                    // ICQ
                    a.push("-global_quality".into());
                    a.push(quality.h26x_qp().to_string());
                }
                RateControl::Bitrate { avg_kbps, max_kbps } => {
                    a.push("-b:v".into());
                    a.push(format!("{avg_kbps}k"));
                    if let Some(m) = max_kbps {
                        a.push("-maxrate".into());
                        a.push(format!("{m}k"));
                    }
                }
            }
        }
        Vendor::Vaapi => {
            // VAAPI has no speed "preset"; -compression_level roughly maps
            // (0 = best quality, 7 = fastest).
            push(&["-compression_level", ["7", "4", "0"][q]]);
            match rate {
                RateControl::Constant(quality) => {
                    push(&["-rc_mode", "CQP", "-qp"]);
                    a.push(hw_const_qp(*quality).to_string());
                }
                RateControl::Bitrate { avg_kbps, max_kbps } => match max_kbps {
                    Some(m) => {
                        push(&["-rc_mode", "VBR"]);
                        a.push("-b:v".into());
                        a.push(format!("{avg_kbps}k"));
                        a.push("-maxrate".into());
                        a.push(format!("{m}k"));
                    }
                    None => {
                        push(&["-rc_mode", "CBR"]);
                        a.push("-b:v".into());
                        a.push(format!("{avg_kbps}k"));
                    }
                },
            }
        }
        Vendor::VideoToolbox => {
            if compression_level == CompressionLevel::Fast {
                push(&["-realtime", "1"]);
            }
            match rate {
                // -q:v support varies by build/codec; falls back gracefully.
                RateControl::Constant(quality) => {
                    a.push("-q:v".into());
                    a.push(quality.videotoolbox_qv().to_string());
                }
                RateControl::Bitrate { avg_kbps, max_kbps } => {
                    a.push("-b:v".into());
                    a.push(format!("{avg_kbps}k"));
                    if let Some(m) = max_kbps {
                        a.push("-maxrate".into());
                        a.push(format!("{m}k"));
                        a.push("-bufsize".into());
                        a.push(format!("{}k", m * 2));
                    }
                }
            }
        }
        Vendor::Software => {
            if codec == Codec::Av1 {
                // libsvtav1: preset 0 (slowest/best) .. 13 (fastest).
                push(&["-preset", ["10", "6", "2"][q]]);
                match rate {
                    RateControl::Constant(quality) => {
                        a.push("-crf".into());
                        a.push(quality.av1_crf().to_string());
                    }
                    RateControl::Bitrate { avg_kbps, .. } => {
                        a.push("-b:v".into());
                        a.push(format!("{avg_kbps}k"));
                    }
                }
            } else {
                // libx264 / libx265
                push(&["-preset", ["veryfast", "medium", "veryslow"][q]]);
                match rate {
                    RateControl::Constant(quality) => {
                        a.push("-crf".into());
                        a.push(quality.h26x_qp().to_string());
                    }
                    RateControl::Bitrate { avg_kbps, max_kbps } => {
                        a.push("-b:v".into());
                        a.push(format!("{avg_kbps}k"));
                        if let Some(m) = max_kbps {
                            a.push("-maxrate".into());
                            a.push(format!("{m}k"));
                            a.push("-bufsize".into());
                            a.push(format!("{}k", m * 2));
                        }
                    }
                }
            }
        }
    }

    a
}

fn select_encoder(codec: Codec, preferred: Option<Vendor>) -> Option<(Vendor, String)> {
    let mut order: Vec<Vendor> = vendor_order().to_vec();
    if let Some(p) = preferred {
        order.retain(|&v| v != p);
        order.insert(0, p);
    }

    for vendor in order {
        if run_probe(codec, vendor) {
            return Some((vendor, encoder_name(codec, vendor)));
        }
    }
    None
}
