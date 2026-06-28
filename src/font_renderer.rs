use std::{cell::LazyCell, sync::LazyLock};

static LIB: LazyLock<freetype::Library> = LazyLock::new(|| freetype::Library::init().unwrap());

pub fn render_fonts_to_atlas() -> ndarray::Array2<u8> {
    todo!()
}
