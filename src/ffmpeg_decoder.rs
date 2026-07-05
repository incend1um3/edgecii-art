use std::collections::HashMap;
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::time::Duration;

use ffmpeg_sidecar::paths::ffmpeg_path;
use image::RgbImage;

use crate::util;

/// A video decoder backed by a spawned FFmpeg process (using the
/// ffmpeg-sidecar-managed binary, same as the encoder). FFmpeg is told to emit
/// raw `rgb24` frames on stdout, which we read one whole frame at a time.
pub struct FfmpegVideoDecoder {
    child: Child,
    stdout: ChildStdout,
    width: u32,
    height: u32,
    fps: f32,
    fps_rational: String,
    frame_size: usize,
    finished: bool,
}

impl FfmpegVideoDecoder {
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn frame_rate(&self) -> f32 {
        self.fps
    }

    pub fn frame_rate_rational(&self) -> &str {
        &self.fps_rational
    }

    /// Read the next raw frame from ffmpeg's stdout.
    ///
    /// Returns `Ok(None)` at a clean end-of-stream (ffmpeg closed the pipe and
    /// exited successfully) and `Err` if ffmpeg died mid-stream (corrupt or
    /// truncated input), so a partial decode can't masquerade as success.
    ///
    /// Because the output is `rawvideo`, every frame is exactly
    /// `width * height * 3` bytes back-to-back, so a short read only ever
    /// happens at EOF on a frame boundary — unless the child died, which the
    /// exit-status check below catches.
    pub fn next_frame(&mut self) -> anyhow::Result<Option<RgbImage>> {
        if self.finished {
            return Ok(None);
        }

        let mut buf = vec![0u8; self.frame_size];
        match self.stdout.read_exact(&mut buf) {
            Ok(()) => Ok(Some(
                RgbImage::from_raw(self.width, self.height, buf).expect("frame size matches dimensions"),
            )),
            Err(e) => {
                self.finished = true;

                // Only wait() without killing when the pipe actually hit EOF:
                // ffmpeg has closed stdout and is about to exit. For any other
                // read error the child may still be alive and blocked writing
                // into the full pipe, so waiting directly would deadlock.
                if e.kind() != ErrorKind::UnexpectedEof {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    return Err(anyhow::anyhow!("error reading frame from ffmpeg: {e}"));
                }

                let status = self.child.wait()?;
                if status.success() {
                    Ok(None)
                } else {
                    Err(anyhow::anyhow!(
                        "ffmpeg decoder exited abnormally ({status}); input is likely corrupt or truncated"
                    ))
                }
            }
        }
    }
}

impl Drop for FfmpegVideoDecoder {
    fn drop(&mut self) {
        // If we stopped early (e.g. the consumer bailed before EOF), don't leave
        // the child running. kill() on an already-exited process is harmless.
        if !matches!(self.child.try_wait(), Ok(Some(_))) {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

pub fn create_decoder(file: &Path) -> anyhow::Result<FfmpegVideoDecoder> {
    if !file.exists() || !file.is_file() {
        anyhow::bail!("Input file not found: {}", file.display());
    }

    let meta = probe_metadata(file)?;

    let hwaccel = probe_hwaccel(file);
    match hwaccel {
        Some(name) => println!("Using hardware accelerated decoding through {}", name),
        None => println!(
            "Warning: using software decoding because no working hardware decoder implementation was found. This will be EXTREMELY slow!"
        ),
    }

    let mut args: Vec<String> = Vec::new();
    args.extend(["-hide_banner", "-v", "error", "-nostdin"].map(String::from));
    if let Some(name) = hwaccel {
        // `-hwaccel X` decodes on the GPU then auto-downloads to system memory
        // (we deliberately don't set `-hwaccel_output_format`, so the rgb24
        // conversion below happens on CPU frames).
        args.extend(["-hwaccel", name].map(String::from));
    }
    // Keep the emitted resolution equal to the coded resolution ffprobe reported
    // (autorotate would swap W/H for rotated phone clips and break the
    // fixed-size frame read).
    args.push("-noautorotate".into());
    args.push("-i".into());
    args.push(file.to_string_lossy().into_owned());
    // Decode only the primary video stream as raw rgb24 to stdout.
    args.extend(["-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"].map(String::from));

    let mut child = Command::new(ffmpeg_path())
        .args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit()) // -v error keeps this quiet unless something breaks
        .spawn()?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("failed to capture ffmpeg stdout"))?;

    Ok(FfmpegVideoDecoder {
        child,
        stdout,
        width: meta.width,
        height: meta.height,
        fps: meta.fps,
        fps_rational: meta.fps_rational,
        frame_size: (meta.width as usize) * (meta.height as usize) * 3,
        finished: false,
    })
}

struct StreamMetadata {
    width: u32,
    height: u32,
    fps: f32,
    /// The raw rational string ffprobe reported for the field `fps` was
    /// derived from, e.g. `30000/1001`.
    fps_rational: String,
}

/// Probe width/height/fps of the primary video stream up front so the caller can
/// size the encoder before any frame has been decoded.
fn probe_metadata(file: &Path) -> anyhow::Result<StreamMetadata> {
    let output = Command::new(ffprobe_path())
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate",
            "-of",
            "default=noprint_wrappers=1",
        ])
        .arg(file)
        .stdin(Stdio::null())
        .output()?;

    anyhow::ensure!(output.status.success(), "ffprobe failed to read {}", file.display());

    let text = String::from_utf8_lossy(&output.stdout);
    // Parse `key=value` lines by name (ffprobe doesn't guarantee field order).
    let fields: HashMap<&str, &str> = text
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim(), v.trim()))
        .collect();

    let width: u32 = fields
        .get("width")
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| anyhow::anyhow!("ffprobe reported no video width for {}", file.display()))?;
    let height: u32 = fields
        .get("height")
        .and_then(|v| v.parse().ok())
        .ok_or_else(|| anyhow::anyhow!("ffprobe reported no video height for {}", file.display()))?;

    // Prefer avg_frame_rate (≈ total_frames/duration, which keeps the constant
    // frame-rate re-encode aligned with the copied audio + `-shortest`), then
    // fall back to the nominal r_frame_rate. Both must parse to a positive,
    // finite rate — streams routinely report `0/0` for one of them.
    let (fps, fps_rational) = ["avg_frame_rate", "r_frame_rate"]
        .iter()
        .find_map(|key| {
            let raw = *fields.get(key)?;
            let fps = parse_rational(raw).filter(|f| f.is_finite() && *f > 0.0)?;
            Some((fps, raw.to_string()))
        })
        .ok_or_else(|| anyhow::anyhow!("ffprobe reported no usable frame rate for {}", file.display()))?;

    Ok(StreamMetadata {
        width,
        height,
        fps,
        fps_rational,
    })
}

fn parse_rational(s: &str) -> Option<f32> {
    let (num, den) = s.split_once('/')?;
    let num: f32 = num.trim().parse().ok()?;
    let den: f32 = den.trim().parse().ok()?;
    (den != 0.0).then_some(num / den)
}

/// Locate the `ffprobe` binary that ffmpeg-sidecar unpacked next to `ffmpeg`.
/// (auto_download installs ffprobe unless `KEEP_ONLY_FFMPEG` is set.)
fn ffprobe_path() -> PathBuf {
    let mut path = ffmpeg_path();
    path.set_file_name(if cfg!(windows) { "ffprobe.exe" } else { "ffprobe" });
    path
}

/// Pick the first hardware decode backend that actually works for this input,
/// mirroring the per-platform priorities the old video-rs path used.
///
/// Simpler alternative: skip this entirely and just pass `-hwaccel auto`, which
/// lets ffmpeg select any available backend and fall back to software on its
/// own — at the cost of the explicit "Using X" / software-warning output.
fn probe_hwaccel(file: &Path) -> Option<&'static str> {
    let candidates: &[&'static str] = if cfg!(target_os = "macos") {
        &["videotoolbox"]
    } else if cfg!(target_os = "windows") {
        &["cuda", "d3d12va", "qsv", "d3d11va", "dxva2"]
    } else {
        &["cuda", "vaapi", "vdpau", "qsv"]
    };

    candidates.iter().copied().find(|name| test_decode(file, name))
}

/// Decode a single frame to the null muxer to confirm a `-hwaccel` can handle
/// this input, with a timeout so a wedged backend can't hang startup.
///
/// `-hwaccel` is best-effort in ffmpeg: when a backend fails to initialize it
/// logs a *warning* ("Failed setup for format ..."), silently falls back to
/// software decoding, and still exits 0. A plain exit-status check would
/// therefore report the first candidate as "working" even on machines where it
/// isn't, so we run at `-v warning`, capture stderr, and treat any hwaccel
/// setup/fallback message as failure.
fn test_decode(file: &Path, hwaccel: &str) -> bool {
    let mut cmd = Command::new(ffmpeg_path());
    cmd.args([
        "-hide_banner",
        "-v",
        "warning",
        "-nostdin",
        "-hwaccel",
        hwaccel,
        "-noautorotate",
        "-i",
    ])
    .arg(file)
    .args(["-map", "0:v:0", "-frames:v", "1", "-f", "null", "-"])
    .stdin(Stdio::null())
    .stdout(Stdio::null())
    .stderr(Stdio::piped());

    let Some((success, stderr)) = util::run_with_timeout_captured(cmd, Duration::from_secs(8)) else {
        return false;
    };
    if !success {
        return false;
    }

    let stderr = stderr.to_lowercase();
    const FALLBACK_MARKERS: &[&str] = &[
        "failed setup for format",               // per-frame hwaccel setup failed → software fallback
        "hwaccel initialization returned error", // same event, second half of the log line
        "device creation failed",                // hw device couldn't even be created
        "error creating a",                      // "Error creating a CUDA/VAAPI/... device" variants
    ];
    !FALLBACK_MARKERS.iter().any(|m| stderr.contains(m))
}
