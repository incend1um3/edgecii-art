use crate::{CHARS, EDGE_CHARS};
use std::sync::LazyLock;

static LIB: LazyLock<freetype::Library> = LazyLock::new(|| freetype::Library::init().unwrap());

pub fn render_fonts_to_atlas(char_height: u32) -> anyhow::Result<(ndarray::Array2<u8>, u32, u32)> {
    let font = include_bytes!("../FiraCodeNerdFontMono-SemiBold.ttf");
    let face = LIB.new_memory_face2(font.as_ref(), 0)?;
    face.set_pixel_sizes(0, char_height)?;

    let size_metrics = face.size_metrics().unwrap();
    let ascender = (size_metrics.ascender >> 6) as i32;
    let cell_w = (size_metrics.max_advance >> 6) as u32;
    let cell_h = (size_metrics.height >> 6) as u32;

    // holds rendered font bitmaps
    let mut char_atlas: ndarray::Array2<u8> =
        ndarray::Array2::zeros((CHARS.len() + EDGE_CHARS.len(), (cell_h * cell_w) as usize));

    // https://freetype.org/freetype2/docs/tutorial/step2.html
    for (i, c) in CHARS.iter().chain(EDGE_CHARS.iter()).enumerate() {
        face.load_char(*c as usize, freetype::face::LoadFlag::RENDER)?;
        let glyph = face.glyph();
        let bitmap = glyph.bitmap();

        let buf = bitmap.buffer();
        let height = bitmap.rows() as i32;
        let width = bitmap.width() as i32;
        let left = glyph.bitmap_left() as i32;
        let top = glyph.bitmap_top() as i32;
        let advance = (glyph.advance().x >> 6) as i32;

        let mut j = 0;
        let y_offset = if ascender > top { ascender - top } else { 0 };
        let x_offset = (cell_w as i32 - advance) / 2;
        for y in 0..height {
            for x in 0..width {
                let ty = y_offset + y;
                let tx = left + x_offset + x;
                if ty >= 0 && (ty as u32) < cell_h && tx >= 0 && (tx as u32) < cell_w {
                    char_atlas[[i, (ty * cell_w as i32 + tx) as usize]] = buf[j];
                }
                j += 1;
            }
        }
    }

    Ok((char_atlas, cell_w, cell_h))
}
