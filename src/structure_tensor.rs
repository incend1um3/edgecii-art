#[derive(Default)]
pub struct CellStructureTensors {
    cell_w: u32,
    cell_h: u32,
    pub tl: StructureTensor,
    pub tr: StructureTensor,
    pub bl: StructureTensor,
    pub br: StructureTensor,
}

impl CellStructureTensors {
    pub fn new(cell_w: u32, cell_h: u32) -> Self {
        assert_eq!(cell_w % 2, 0);
        assert_eq!(cell_h % 2, 0);

        let subcell_w = cell_w / 2;
        let subcell_h = cell_h / 2;
        let quadrant_pixels = subcell_w * subcell_h;

        Self {
            cell_w,
            cell_h,
            tl: StructureTensor::new(quadrant_pixels),
            tr: StructureTensor::new(quadrant_pixels),
            bl: StructureTensor::new(quadrant_pixels),
            br: StructureTensor::new(quadrant_pixels),
        }
    }

    #[inline]
    pub fn accumulate(&mut self, dx: u32, dy: u32, gx: f32, gy: f32) {
        if dx < self.cell_w / 2 {
            if dy < self.cell_h / 2 {
                self.tl.accumulate(gx, gy);
            } else {
                self.bl.accumulate(gx, gy);
            }
        } else {
            if dy < self.cell_h / 2 {
                self.tr.accumulate(gx, gy);
            } else {
                self.br.accumulate(gx, gy);
            }
        }
    }

    #[inline(always)]
    pub fn combined(&self) -> StructureTensor {
        self.tl.combine(&self.bl).combine(&self.tr).combine(&self.br)
    }

    #[inline]
    pub fn left(&self) -> StructureTensor {
        self.tl.combine(&self.bl)
    }

    #[inline]
    pub fn right(&self) -> StructureTensor {
        self.tr.combine(&self.br)
    }

    #[inline]
    pub fn top(&self) -> StructureTensor {
        self.tl.combine(&self.tr)
    }

    #[inline]
    pub fn bottom(&self) -> StructureTensor {
        self.bl.combine(&self.br)
    }
}

#[derive(Default)]
pub struct StructureTensor {
    pixels: u32,
    /// ∑ gx^2
    gx_squared: f32,
    /// ∑ gxgy
    gxgy: f32,
    /// ∑ gy^2
    gy_squared: f32,
}

impl StructureTensor {
    pub(super) fn new(pixels: u32) -> Self {
        Self {
            pixels,
            ..Default::default()
        }
    }

    pub fn combine(&self, other: &Self) -> Self {
        Self {
            pixels: self.pixels + other.pixels,
            gx_squared: self.gx_squared + other.gx_squared,
            gxgy: self.gxgy + other.gxgy,
            gy_squared: self.gy_squared + other.gy_squared,
        }
    }

    #[inline]
    pub fn accumulate(&mut self, gx: f32, gy: f32) {
        self.gx_squared += gx * gx;
        self.gxgy += gx * gy;
        self.gy_squared += gy * gy;
    }

    /// eigendecomposition
    /// tan 2θ = 2b / (a - c)
    #[inline]
    pub fn theta(&self) -> f32 {
        0.5 * f32::atan2(2.0 * self.gxgy, self.gx_squared - self.gy_squared)
    }

    /// Sum of the two eigenvalues
    #[inline(always)]
    pub fn trace(&self) -> f32 {
        self.gx_squared + self.gy_squared
    }

    // (λ₁ - λ₂) / (λ₁ + λ₂)
    #[inline]
    pub fn coherence(&self) -> f32 {
        let sub = self.gx_squared - self.gy_squared;
        sub.mul_add(sub, 4.0 * self.gxgy * self.gxgy).sqrt() / self.trace()
    }

    #[inline]
    pub fn energy_avg(&self) -> f32 {
        self.trace() / (self.pixels) as f32
    }

    /// Rayleigh quotient along the unit vector given by theta
    #[inline]
    pub fn directional_energy(&self, theta: f32) -> f32 {
        let (s, c) = theta.sin_cos();
        self.gx_squared * c * c + 2.0 * self.gxgy * c * s + self.gy_squared * s * s
    }

    #[inline]
    pub fn directional_alignment(&self, theta: f32) -> f32 {
        let e = self.directional_energy(theta);
        (2.0 * e - self.trace()) / self.trace()
    }
}
