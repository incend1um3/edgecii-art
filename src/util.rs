use image::RgbImage;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

pub const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

pub fn image_to_frame(img: RgbImage) -> ndarray::Array3<u8> {
    let (w, h) = img.dimensions();
    ndarray::Array3::from_shape_vec((h as usize, w as usize, 3), img.into_raw())
        .expect("buffer size matches dimensions")
}

/// Run a command to completion (or kill it at `timeout`), returning whether it
/// exited successfully plus everything it wrote to stderr. Returns `None` on
/// spawn failure or timeout. Sets stderr to piped; it's drained on a separate
/// thread so a chatty child can't deadlock against a full pipe.
pub fn run_with_timeout_captured(mut cmd: Command, timeout: Duration) -> Option<(bool, String)> {
    cmd.stderr(Stdio::piped());
    let mut child = cmd.spawn().ok()?;
    let mut stderr = child.stderr.take()?;
    let reader = std::thread::spawn(move || {
        let mut buf = String::new();
        let _ = stderr.read_to_string(&mut buf);
        buf
    });

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stderr = reader.join().unwrap_or_default();
                return Some((status.success(), stderr));
            }
            Ok(None) if start.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                return None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader.join();
                return None;
            }
        }
    }
}

/// Success/failure only; stderr is discarded.
pub fn run_with_timeout(cmd: Command, timeout: Duration) -> bool {
    matches!(run_with_timeout_captured(cmd, timeout), Some((true, _)))
}
