#[derive(Default)]
pub struct CellStructureTensors {
    cell_w: u32,
    cell_h: u32,
    pub top: StructureTensor,
    pub bottom: StructureTensor,
    pub left: StructureTensor,
    pub right: StructureTensor,
    pub combined: StructureTensor,
}

impl CellStructureTensors {
    pub fn new(cell_w: u32, cell_h: u32) -> Self {
        assert_eq!(cell_w % 2, 0);
        assert_eq!(cell_h % 2, 0);

        let subcell_w = cell_w / 2;
        let subcell_h = cell_h / 2;

        Self {
            cell_w,
            cell_h,
            top: StructureTensor::new(cell_w, subcell_h),
            bottom: StructureTensor::new(cell_w, subcell_h),
            left: StructureTensor::new(subcell_w, cell_h),
            right: StructureTensor::new(subcell_w, cell_h),
            combined: StructureTensor::new(cell_w, cell_h),
        }
    }

    #[inline]
    pub fn accumulate(&mut self, dx: u32, dy: u32, gx: f32, gy: f32) {
        if dx < self.cell_w / 2 {
            self.left.accumulate(gx, gy);
        } else {
            self.right.accumulate(gx, gy);
        }

        if dy < self.cell_h / 2 {
            self.top.accumulate(gx, gy);
        } else {
            self.bottom.accumulate(gx, gy);
        }

        self.combined.accumulate(gx, gy);
    }
}

#[derive(Default)]
pub struct StructureTensor {
    cell_w: u32,
    cell_h: u32,

    /// ∑ gx^2
    gx_squared: f32,
    /// ∑ gxgy
    gxgy: f32,
    /// ∑ gy^2
    gy_squared: f32,
}

impl StructureTensor {
    pub(super) fn new(cell_w: u32, cell_h: u32) -> Self {
        Self {
            cell_w,
            cell_h,
            ..Default::default()
        }
    }

    #[inline]
    pub fn accumulate(&mut self, gx: f32, gy: f32) {
        self.gx_squared += gx * gx;
        self.gxgy += gx * gy;
        self.gy_squared += gy * gy;
    }

    #[inline]
    pub fn theta(&self) -> f32 {
        // eigendecomposition
        // tan 2θ = 2b / (a - c)
        0.5 * libm::atan2f(2.0 * self.gxgy, self.gx_squared - self.gy_squared)
    }

    /// Sum of the two eigenvalues
    #[inline(always)]
    pub fn trace(&self) -> f32 {
        self.gx_squared + self.gy_squared
    }

    #[inline]
    pub fn coherence(&self) -> f32 {
        ((self.gx_squared - self.gy_squared).powi(2) + 4.0 * self.gxgy * self.gxgy).sqrt()
            / self.trace()
    }

    #[inline]
    pub fn energy_avg(&self) -> f32 {
        self.trace() / (self.cell_h * self.cell_w) as f32
    }
}
