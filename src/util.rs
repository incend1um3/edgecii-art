use image::{ImageBuffer, Luma, RgbImage};

#[macro_export]
macro_rules! join3 {
    ($a:expr, $b:expr, $c:expr) => {{
        let (a, (b, c)) = rayon::join($a, || rayon::join($b, $c));
        (a, b, c)
    }};
}

pub fn luma_f32_to_u8(img: &ImageBuffer<Luma<f32>, Vec<f32>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let (width, height) = img.dimensions();

    ImageBuffer::from_fn(width, height, |x, y| {
        let Luma([v]) = *img.get_pixel(x, y);
        Luma([(v.clamp(0.0, 1.0) * 255.0).round() as u8])
    })
}

pub fn video_frame_to_image(frame: &ndarray::Array3<u8>) -> RgbImage {
    let (h, w, _) = frame.dim();
    // Ensure contiguous, standard (row-major) layout before grabbing the buffer.
    let frame = frame.as_standard_layout();
    let (raw, _) = frame.to_owned().into_raw_vec_and_offset();
    RgbImage::from_raw(w as u32, h as u32, raw).expect("buffer size matches dimensions")
}

pub fn image_to_frame(img: &RgbImage) -> ndarray::Array3<u8> {
    let (w, h) = img.dimensions();
    ndarray::Array3::from_shape_vec((h as usize, w as usize, 3), img.as_raw().clone())
        .expect("buffer size matches dimensions")
}
