use image::{ImageBuffer, Luma};

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
