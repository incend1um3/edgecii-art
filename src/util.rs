use image::RgbImage;

#[macro_export]
macro_rules! join3 {
    ($a:expr, $b:expr, $c:expr) => {{
        let (a, (b, c)) = rayon::join($a, || rayon::join($b, $c));
        (a, b, c)
    }};
}

pub fn video_frame_to_image(frame: ndarray::Array3<u8>) -> RgbImage {
    let (h, w, _) = frame.dim();

    let raw = if frame.is_standard_layout() {
        // Zero-copy
        frame.into_raw_vec_and_offset().0
    } else {
        // Non-contiguous
        frame
            .as_standard_layout()
            .into_owned()
            .into_raw_vec_and_offset()
            .0
    };

    RgbImage::from_raw(w as u32, h as u32, raw).expect("buffer size matches dimensions")
}

pub fn image_to_frame(img: &RgbImage) -> ndarray::Array3<u8> {
    let (w, h) = img.dimensions();
    ndarray::Array3::from_shape_vec((h as usize, w as usize, 3), img.as_raw().clone())
        .expect("buffer size matches dimensions")
}
