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

/// Trait for types that support tolerance-based numerical comparisons.
///
/// This trait provides the operations needed for tolerance-aware pivot checks
/// in PLU decomposition and other numerical algorithms. It abstracts over both
/// real (f32, f64) and complex (Complex<f32>, Complex<f64>) floating-point types,
/// as well as exact arithmetic types (integers).
///
/// # Tolerance-Based Pivoting
///
/// In exact arithmetic (integers), we can check if a pivot is exactly zero.
/// In floating-point arithmetic, this is unreliable due to:
/// - Rounding errors from previous operations
/// - Loss of significance in subtraction
/// - Accumulation of numerical errors
///
/// Instead, we use **tolerance-based pivoting**: treat a pivot as zero if its
/// magnitude is "too small" relative to the matrix scale. This prevents:
/// - Division by very small numbers (amplifies rounding errors)
/// - False invertibility detection for nearly singular matrices
/// - Numerical instability in subsequent elimination steps
pub trait ToleranceOps {
    /// The type returned by absolute value (same as Self for reals, f32/f64 for Complex)
    /// This type must support comparison operations for tolerance checking.
    type Abs: PartialOrd;

    /// Compute the absolute value (magnitude) of this element.
    /// For real numbers: |x|. For complex numbers: sqrt(re^2 + im^2).
    fn abs_val(&self) -> Self::Abs;

    /// Machine epsilon: smallest value ε such that 1.0 + ε != 1.0
    /// This represents the unit roundoff error in floating-point arithmetic.
    ///
    /// For exact types (integers), this returns None to signal exact arithmetic.
    /// For f32: approximately 1.19e-7
    /// For f64: approximately 2.22e-16
    fn epsilon() -> Option<Self::Abs>;
}

// Implementations for floating-point types (inexact arithmetic)

impl ToleranceOps for f32 {
    type Abs = f32;

    fn abs_val(&self) -> Self::Abs {
        self.abs()
    }

    fn epsilon() -> Option<Self::Abs> {
        Some(f32::EPSILON)
    }
}

impl ToleranceOps for f64 {
    type Abs = f64;

    fn abs_val(&self) -> Self::Abs {
        self.abs()
    }

    fn epsilon() -> Option<Self::Abs> {
        Some(f64::EPSILON)
    }
}

impl ToleranceOps for Complex<f32> {
    type Abs = f32;

    fn abs_val(&self) -> Self::Abs {
        self.norm() // sqrt(re^2 + im^2)
    }

    fn epsilon() -> Option<Self::Abs> {
        Some(f32::EPSILON)
    }
}

impl ToleranceOps for Complex<f64> {
    type Abs = f64;

    fn abs_val(&self) -> Self::Abs {
        self.norm() // sqrt(re^2 + im^2)
    }

    fn epsilon() -> Option<Self::Abs> {
        Some(f64::EPSILON)
    }
}

// Implementations for integer types (exact arithmetic)
// epsilon() returns None to signal that no tolerance is needed

impl ToleranceOps for i8 {
    type Abs = i8;

    fn abs_val(&self) -> Self::Abs {
        self.abs()
    }

    fn epsilon() -> Option<Self::Abs> {
        None // Exact arithmetic
    }
}

impl ToleranceOps for i16 {
    type Abs = i16;

    fn abs_val(&self) -> Self::Abs {
        self.abs()
    }

    fn epsilon() -> Option<Self::Abs> {
        None // Exact arithmetic
    }
}

impl ToleranceOps for i32 {
    type Abs = i32;

    fn abs_val(&self) -> Self::Abs {
        self.abs()
    }

    fn epsilon() -> Option<Self::Abs> {
        None // Exact arithmetic
    }
}

impl ToleranceOps for i64 {
    type Abs = i64;

    fn abs_val(&self) -> Self::Abs {
        self.abs()
    }

    fn epsilon() -> Option<Self::Abs> {
        None // Exact arithmetic
    }
}

impl ToleranceOps for i128 {
    type Abs = i128;

    fn abs_val(&self) -> Self::Abs {
        self.abs()
    }

    fn epsilon() -> Option<Self::Abs> {
        None // Exact arithmetic
    }
}

impl ToleranceOps for isize {
    type Abs = isize;

    fn abs_val(&self) -> Self::Abs {
        self.abs()
    }

    fn epsilon() -> Option<Self::Abs> {
        None // Exact arithmetic
    }
}

impl ToleranceOps for u8 {
    type Abs = u8;

    fn abs_val(&self) -> Self::Abs {
        *self
    }

    fn epsilon() -> Option<Self::Abs> {
        None // Exact arithmetic
    }
}

impl ToleranceOps for u16 {
    type Abs = u16;

    fn abs_val(&self) -> Self::Abs {
        *self
    }

    fn epsilon() -> Option<Self::Abs> {
        None // Exact arithmetic
    }
}

impl ToleranceOps for u32 {
    type Abs = u32;

    fn abs_val(&self) -> Self::Abs {
        *self
    }

    fn epsilon() -> Option<Self::Abs> {
        None // Exact arithmetic
    }
}

impl ToleranceOps for u64 {
    type Abs = u64;

    fn abs_val(&self) -> Self::Abs {
        *self
    }

    fn epsilon() -> Option<Self::Abs> {
        None // Exact arithmetic
    }
}

impl ToleranceOps for u128 {
    type Abs = u128;

    fn abs_val(&self) -> Self::Abs {
        *self
    }

    fn epsilon() -> Option<Self::Abs> {
        None // Exact arithmetic
    }
}

impl ToleranceOps for usize {
    type Abs = usize;

    fn abs_val(&self) -> Self::Abs {
        *self
    }

    fn epsilon() -> Option<Self::Abs> {
        None // Exact arithmetic
    }
}
