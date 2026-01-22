//! Trait for magnitude comparison used in partial pivoting.

use num_complex::Complex;

/// Trait for types that support magnitude comparison for pivoting.
/// This is needed to find the largest absolute value element for partial pivoting.
pub trait PivotOrd: Sized {
    /// The type of the comparison key (e.g., Self for reals, f32/f64 for Complex)
    type Key: PartialOrd;

    /// Return a value suitable for pivot comparison (e.g., absolute value for signed types)
    fn pivot_key(&self) -> Self::Key;
}

// For unsigned integers, the pivot key is the value itself
impl PivotOrd for u8 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        *self
    }
}

impl PivotOrd for u16 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        *self
    }
}

impl PivotOrd for u32 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        *self
    }
}

impl PivotOrd for u64 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        *self
    }
}

impl PivotOrd for u128 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        *self
    }
}

impl PivotOrd for usize {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        *self
    }
}

// For signed integers, the pivot key is the absolute value
impl PivotOrd for i8 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        self.saturating_abs()
    }
}

impl PivotOrd for i16 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        self.saturating_abs()
    }
}

impl PivotOrd for i32 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        self.saturating_abs()
    }
}

impl PivotOrd for i64 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        self.saturating_abs()
    }
}

impl PivotOrd for i128 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        self.saturating_abs()
    }
}

impl PivotOrd for isize {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        self.saturating_abs()
    }
}

// For floating point, use absolute value
impl PivotOrd for f32 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        self.abs()
    }
}

impl PivotOrd for f64 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        self.abs()
    }
}

// For complex numbers, use magnitude (norm) as the key
impl PivotOrd for Complex<f32> {
    type Key = f32;

    fn pivot_key(&self) -> Self::Key {
        self.norm()
    }
}

impl PivotOrd for Complex<f64> {
    type Key = f64;

    fn pivot_key(&self) -> Self::Key {
        self.norm()
    }
}
