/// Trait for types that can be used as matrix elements.
///
/// This trait defines the minimal operations required for a type to be used
/// in a matrix. It requires the type to be clonable and comparable, and to
/// provide additive and multiplicative identity elements.
pub trait MatrixElement: Clone + PartialEq {
    /// Return the additive identity (zero) for this type.
    fn zero() -> Self;

    /// Return the multiplicative identity (one) for this type.
    fn one() -> Self;
}

// Floating point implementations
impl MatrixElement for f32 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }
}

impl MatrixElement for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }
}

// Signed integer implementations
impl MatrixElement for i8 {
    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }
}

impl MatrixElement for i16 {
    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }
}

impl MatrixElement for i32 {
    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }
}

impl MatrixElement for i64 {
    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }
}

impl MatrixElement for i128 {
    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }
}

impl MatrixElement for isize {
    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }
}

// Unsigned integer implementations
impl MatrixElement for u8 {
    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }
}

impl MatrixElement for u16 {
    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }
}

impl MatrixElement for u32 {
    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }
}

impl MatrixElement for u64 {
    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }
}

impl MatrixElement for u128 {
    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }
}

impl MatrixElement for usize {
    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }
}

// Complex number implementation
use num_complex::Complex;

impl MatrixElement for Complex<f32> {
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }

    fn one() -> Self {
        Complex::new(1.0, 0.0)
    }
}

impl MatrixElement for Complex<f64> {
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }

    fn one() -> Self {
        Complex::new(1.0, 0.0)
    }
}
