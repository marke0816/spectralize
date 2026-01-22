//! Traits for absolute value and square root operations used in matrix norms.

use num_complex::Complex;

/// Trait for types that support absolute value operations.
/// Required for computing matrix norms that depend on element magnitudes.
pub trait Abs {
    /// The output type of the absolute value operation.
    /// For real numbers, this is Self. For complex numbers, this is the underlying real type.
    type Output;

    /// Compute the absolute value (magnitude) of this element.
    fn abs(self) -> Self::Output;
}

/// Trait for types that support square root operations.
/// Required for computing the Frobenius norm.
pub trait Sqrt {
    /// Compute the square root of this value.
    fn sqrt(self) -> Self;
}

// Implementations for f32
impl Abs for f32 {
    type Output = f32;

    fn abs(self) -> Self::Output {
        f32::abs(self)
    }
}

impl Sqrt for f32 {
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}

// Implementations for f64
impl Abs for f64 {
    type Output = f64;

    fn abs(self) -> Self::Output {
        f64::abs(self)
    }
}

impl Sqrt for f64 {
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

// Implementations for Complex<f32>
impl Abs for Complex<f32> {
    type Output = f32;

    fn abs(self) -> Self::Output {
        self.norm()
    }
}

// Implementations for Complex<f64>
impl Abs for Complex<f64> {
    type Output = f64;

    fn abs(self) -> Self::Output {
        self.norm()
    }
}

// Implementations for signed integer types
impl Abs for i8 {
    type Output = i8;

    fn abs(self) -> Self::Output {
        i8::saturating_abs(self)
    }
}

impl Abs for i16 {
    type Output = i16;

    fn abs(self) -> Self::Output {
        i16::saturating_abs(self)
    }
}

impl Abs for i32 {
    type Output = i32;

    fn abs(self) -> Self::Output {
        i32::saturating_abs(self)
    }
}

impl Abs for i64 {
    type Output = i64;

    fn abs(self) -> Self::Output {
        i64::saturating_abs(self)
    }
}

impl Abs for i128 {
    type Output = i128;

    fn abs(self) -> Self::Output {
        i128::saturating_abs(self)
    }
}

impl Abs for isize {
    type Output = isize;

    fn abs(self) -> Self::Output {
        isize::saturating_abs(self)
    }
}

// Implementations for unsigned integer types (already absolute)
impl Abs for u8 {
    type Output = u8;

    fn abs(self) -> Self::Output {
        self
    }
}

impl Abs for u16 {
    type Output = u16;

    fn abs(self) -> Self::Output {
        self
    }
}

impl Abs for u32 {
    type Output = u32;

    fn abs(self) -> Self::Output {
        self
    }
}

impl Abs for u64 {
    type Output = u64;

    fn abs(self) -> Self::Output {
        self
    }
}

impl Abs for u128 {
    type Output = u128;

    fn abs(self) -> Self::Output {
        self
    }
}

impl Abs for usize {
    type Output = usize;

    fn abs(self) -> Self::Output {
        self
    }
}
