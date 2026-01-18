//! Compile-time dimension matrices using const generics.
//!
//! This module provides `ConstMatrix<T, R, C>` where dimensions are known at compile time,
//! enabling the type system to enforce dimension compatibility for operations.
//!
//! # Examples
//!
//! ```
//! use spectralize::matrix::{ConstMatrix, Matrix};
//!
//! // Create a 3x3 identity matrix - dimensions are in the type
//! let identity: ConstMatrix<f64, 3, 3> = ConstMatrix::identity();
//!
//! // Addition requires matching dimensions (enforced at compile time)
//! let a: ConstMatrix<f64, 2, 2> = ConstMatrix::identity();
//! let b: ConstMatrix<f64, 2, 2> = ConstMatrix::identity();
//! let c = a + b; // ✓ Compiles
//!
//! // Multiplication enforces inner dimension compatibility
//! let d: ConstMatrix<f64, 2, 3> = ConstMatrix::zero();
//! let e: ConstMatrix<f64, 3, 4> = ConstMatrix::zero();
//! let f: ConstMatrix<f64, 2, 4> = d * e; // ✓ Compiles - dimensions match
//!
//! // Convert to/from dynamic Matrix
//! let dyn_mat: Matrix<f64> = identity.into(); // Always succeeds
//! let const_mat: ConstMatrix<f64, 3, 3> = dyn_mat.try_into()?; // Fallible
//! # Ok::<(), spectralize::matrix::MatrixError>(())
//! ```

use super::norm::Sqrt;
use super::{Matrix, MatrixElement, MatrixError};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A matrix with compile-time known dimensions.
///
/// Unlike `Matrix<T>` where dimensions are runtime values, `ConstMatrix<T, R, C>`
/// encodes dimensions in the type system. This provides:
/// - Compile-time dimension compatibility checking
/// - No runtime dimension validation overhead
/// - Type-safe operations that cannot fail due to dimension mismatches
///
/// # Storage
///
/// Uses `Vec<T>` in row-major order with the invariant `data.len() == R * C`.
/// While heap-allocated, this avoids stack overflow for large matrices and
/// works with non-Copy types (like `Complex<f64>`).
///
/// # Type Parameters
///
/// - `T`: Element type, must implement `MatrixElement + Debug`
/// - `R`: Number of rows (compile-time constant)
/// - `C`: Number of columns (compile-time constant)
#[derive(Debug, Clone, PartialEq)]
pub struct ConstMatrix<T: MatrixElement + std::fmt::Debug, const R: usize, const C: usize> {
    data: Vec<T>,
}

fn scaled_tolerance<T, const N: usize>(data: &[T]) -> T::Abs
where
    T: super::ToleranceOps,
    T::Abs: MatrixElement + Add<Output = T::Abs> + Mul<Output = T::Abs> + PartialOrd,
{
    match T::epsilon() {
        Some(eps) => {
            let mut max_sum = T::Abs::zero();
            for row in 0..N {
                let mut sum = T::Abs::zero();
                for col in 0..N {
                    sum = sum + data[row * N + col].abs_val();
                }
                if sum > max_sum {
                    max_sum = sum;
                }
            }

            let mut n_factor = T::Abs::one();
            for _ in 1..N {
                n_factor = n_factor + T::Abs::one();
            }

            n_factor * (eps * max_sum)
        }
        None => T::Abs::zero(),
    }
}

impl<T: MatrixElement + std::fmt::Debug, const R: usize, const C: usize> ConstMatrix<T, R, C> {
    /// Create a new matrix from a `Vec<T>`.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != R * C`.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![
    ///     1, 2,
    ///     3, 4,
    /// ]);
    /// assert_eq!(m.get(0, 1), 2);
    /// ```
    pub fn new(data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            R * C,
            "Data length {} does not match dimensions {}x{} (expected {})",
            data.len(),
            R,
            C,
            R * C
        );
        Self { data }
    }

    /// Create a matrix filled with zeros.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<f64, 3, 3> = ConstMatrix::zero();
    /// assert_eq!(m.get(1, 1), 0.0);
    /// ```
    pub fn zero() -> Self {
        Self {
            data: vec![T::zero(); R * C],
        }
    }

    /// Create an identity matrix.
    ///
    /// For non-square matrices (R != C), creates a rectangular identity matrix
    /// with ones on the main diagonal and zeros elsewhere.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<i32, 3, 3> = ConstMatrix::identity();
    /// assert_eq!(m.get(0, 0), 1);
    /// assert_eq!(m.get(0, 1), 0);
    ///
    /// // Rectangular identity
    /// let rect: ConstMatrix<i32, 2, 3> = ConstMatrix::identity();
    /// assert_eq!(rect.get(0, 0), 1);
    /// assert_eq!(rect.get(1, 1), 1);
    /// assert_eq!(rect.get(1, 2), 0);
    /// ```
    pub fn identity() -> Self {
        let mut data = vec![T::zero(); R * C];
        for i in 0..R.min(C) {
            data[i * C + i] = T::one();
        }
        Self { data }
    }

    /// Get the number of rows (always returns `R`).
    ///
    /// This method exists for API compatibility with `Matrix<T>`.
    /// The value is known at compile time from the type.
    #[inline]
    pub const fn rows(&self) -> usize {
        R
    }

    /// Get the number of columns (always returns `C`).
    ///
    /// This method exists for API compatibility with `Matrix<T>`.
    /// The value is known at compile time from the type.
    #[inline]
    pub const fn cols(&self) -> usize {
        C
    }

    /// Get a reference to an element without cloning.
    ///
    /// # Panics
    ///
    /// Panics if `row >= R` or `col >= C`.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    /// assert_eq!(*m.get_ref(0, 1), 2);
    /// ```
    #[inline]
    pub fn get_ref(&self, row: usize, col: usize) -> &T {
        assert!(row < R && col < C, "Index out of bounds");
        &self.data[row * C + col]
    }

    /// Get a clone of an element.
    ///
    /// # Panics
    ///
    /// Panics if `row >= R` or `col >= C`.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    /// assert_eq!(m.get(1, 0), 3);
    /// ```
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> T {
        assert!(row < R && col < C, "Index out of bounds");
        self.data[row * C + col].clone()
    }

    /// Set the value of an element.
    ///
    /// # Panics
    ///
    /// Panics if `row >= R` or `col >= C`.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let mut m: ConstMatrix<i32, 2, 2> = ConstMatrix::zero();
    /// m.set(0, 1, 5);
    /// assert_eq!(m.get(0, 1), 5);
    /// ```
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(row < R && col < C, "Index out of bounds");
        self.data[row * C + col] = value;
    }

    /// Get a slice containing a row.
    ///
    /// # Panics
    ///
    /// Panics if `row >= R`.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<i32, 2, 3> = ConstMatrix::new(vec![1, 2, 3, 4, 5, 6]);
    /// assert_eq!(m.row(0), &[1, 2, 3]);
    /// assert_eq!(m.row(1), &[4, 5, 6]);
    /// ```
    pub fn row(&self, row: usize) -> &[T] {
        assert!(row < R, "Row index out of bounds");
        let start = row * C;
        &self.data[start..start + C]
    }

    /// Iterator over a column without allocation.
    ///
    /// Returns an iterator that yields references to elements in the specified column.
    ///
    /// # Panics
    ///
    /// Panics if `col >= C`.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<i32, 3, 2> = ConstMatrix::new(vec![1, 2, 3, 4, 5, 6]);
    /// let col: Vec<i32> = m.col_iter(1).cloned().collect();
    /// assert_eq!(col, vec![2, 4, 6]);
    /// ```
    pub fn col_iter(&self, col: usize) -> impl Iterator<Item = &T> {
        assert!(col < C, "Column index out of bounds");
        (0..R).map(move |r| &self.data[r * C + col])
    }

    /// Transpose the matrix, swapping rows and columns.
    ///
    /// Returns a new `ConstMatrix<T, C, R>` where dimensions are swapped.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<i32, 2, 3> = ConstMatrix::new(vec![
    ///     1, 2, 3,
    ///     4, 5, 6,
    /// ]);
    /// let mt: ConstMatrix<i32, 3, 2> = m.transpose();
    /// assert_eq!(mt.get(0, 0), 1);
    /// assert_eq!(mt.get(0, 1), 4);
    /// assert_eq!(mt.get(1, 0), 2);
    /// ```
    pub fn transpose(&self) -> ConstMatrix<T, C, R> {
        let mut data = Vec::with_capacity(R * C);

        // Iterate column-major to populate transposed row-major layout
        for col in 0..C {
            for row in 0..R {
                data.push(self.data[row * C + col].clone());
            }
        }

        ConstMatrix { data }
    }

    /// Get a reference to the underlying data vector.
    ///
    /// The data is stored in row-major order.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Consume the matrix and return the underlying data vector.
    ///
    /// The data is in row-major order with length `R * C`.
    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    /// Compute the trace (sum of diagonal elements) of a square matrix.
    ///
    /// Only available for square matrices where R == C.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<i32, 3, 3> = ConstMatrix::new(vec![
    ///     1, 2, 3,
    ///     4, 5, 6,
    ///     7, 8, 9,
    /// ]);
    /// assert_eq!(m.trace(), 15); // 1 + 5 + 9
    /// ```
    pub fn trace(&self) -> T
    where
        T: Add<Output = T>,
    {
        // Note: This compiles for all ConstMatrix<T, R, C>, but is most meaningful when R == C.
        // For non-square matrices, it sums the min(R, C) diagonal elements.
        (0..R.min(C))
            .map(|i| self.data[i * C + i].clone())
            .fold(T::zero(), |acc, x| acc + x)
    }

    /// Dot product (inner product) treating the matrix as a vector.
    ///
    /// Computes the sum of element-wise products: sum(self[i] * other[i]).
    /// Both matrices must have the same dimensions (R and C).
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    /// let b: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![5, 6, 7, 8]);
    /// assert_eq!(a.dot(&b), 70); // 1*5 + 2*6 + 3*7 + 4*8
    /// ```
    pub fn dot(&self, other: &ConstMatrix<T, R, C>) -> T
    where
        T: Mul<Output = T> + Add<Output = T>,
    {
        let mut result = T::zero();
        for i in 0..self.data.len() {
            result = result + (self.data[i].clone() * other.data[i].clone());
        }
        result
    }

    /// Outer product treating matrices as vectors.
    ///
    /// For self (length R*C) and other (length R2*C2), produces an (R*C)×(R2*C2) matrix
    /// where result[i,j] = self[i] * other[j].
    ///
    /// Returns a dynamic `Matrix<T>` since the result dimensions involve const arithmetic
    /// which is not stable in Rust yet.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::{ConstMatrix, Matrix};
    ///
    /// let a: ConstMatrix<i32, 2, 1> = ConstMatrix::new(vec![1, 2]);
    /// let b: ConstMatrix<i32, 3, 1> = ConstMatrix::new(vec![3, 4, 5]);
    /// let c: Matrix<i32> = a.outer(&b);
    /// assert_eq!(c.get(0, 0), 3);  // 1 * 3
    /// assert_eq!(c.get(0, 2), 5);  // 1 * 5
    /// assert_eq!(c.get(1, 1), 8);  // 2 * 4
    /// ```
    pub fn outer<const R2: usize, const C2: usize>(
        &self,
        other: &ConstMatrix<T, R2, C2>,
    ) -> Matrix<T>
    where
        T: Mul<Output = T>,
    {
        let m = self.data.len();
        let n = other.data.len();

        let mut data = Vec::with_capacity(m * n);

        for i in 0..m {
            for j in 0..n {
                data.push(self.data[i].clone() * other.data[j].clone());
            }
        }

        Matrix::new(m, n, data)
    }

    /// Matrix exponentiation: raises a square matrix to a non-negative integer power.
    ///
    /// Uses binary exponentiation for O(log n) complexity.
    /// - A^0 = I (identity matrix)
    /// - A^1 = A
    /// - A^n = A * A * ... * A (n times)
    ///
    /// Note: This method is available for all matrices, but is only mathematically valid
    /// for square matrices (R == C).
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    /// let a_squared: ConstMatrix<i32, 2, 2> = a.pow(2);
    /// assert_eq!(a_squared.get(0, 0), 7);  // Result of A * A
    /// assert_eq!(a_squared.get(1, 1), 22);
    /// ```
    pub fn pow(&self, n: i32) -> Self
    where
        T: Mul<Output = T> + Add<Output = T>,
    {
        // Const matrices only support non-negative exponents
        // (negative exponents would require types that support inversion)
        assert!(
            n >= 0,
            "ConstMatrix pow only supports non-negative exponents. \
             Use dynamic Matrix with f32/f64 for negative exponents."
        );

        let n_u32 = n as u32;
        match n_u32 {
            0 => Self::identity(),
            1 => self.clone(),
            _ => {
                // Convert to dynamic matrix, use its pow, convert back
                // This avoids implementing binary exponentiation twice
                let dyn_self: Matrix<T> = self.clone().into();
                // Use binary exponentiation directly to avoid calling inverse-requiring pow
                let mut result = Matrix::identity(R, C);
                let mut base = dyn_self;
                let mut exp = n_u32;

                while exp > 0 {
                    if exp % 2 == 1 {
                        result = result * &base;
                    }
                    base = &base * &base;
                    exp /= 2;
                }

                // Convert back to ConstMatrix
                let mut data = Vec::with_capacity(R * C);
                for row in 0..R {
                    for col in 0..C {
                        data.push(result.get(row, col));
                    }
                }
                Self { data }
            }
        }
    }
}

// Cross product for 3D vectors (specific implementation for 3x1 and 1x3 matrices)
impl<T> ConstMatrix<T, 3, 1>
where
    T: MatrixElement + std::fmt::Debug + Mul<Output = T> + Sub<Output = T>,
{
    /// Cross product of two 3D column vectors.
    ///
    /// Computes the vector cross product a × b = (a₂b₃ - a₃b₂, a₃b₁ - a₁b₃, a₁b₂ - a₂b₁).
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let a: ConstMatrix<i32, 3, 1> = ConstMatrix::new(vec![1, 0, 0]);
    /// let b: ConstMatrix<i32, 3, 1> = ConstMatrix::new(vec![0, 1, 0]);
    /// let c = a.cross(&b);
    /// assert_eq!(c.get(0, 0), 0);
    /// assert_eq!(c.get(1, 0), 0);
    /// assert_eq!(c.get(2, 0), 1); // i × j = k
    /// ```
    pub fn cross(&self, other: &ConstMatrix<T, 3, 1>) -> ConstMatrix<T, 3, 1> {
        let a1 = &self.data[0];
        let a2 = &self.data[1];
        let a3 = &self.data[2];

        let b1 = &other.data[0];
        let b2 = &other.data[1];
        let b3 = &other.data[2];

        let mut data = Vec::with_capacity(3);
        data.push(a2.clone() * b3.clone() - a3.clone() * b2.clone());
        data.push(a3.clone() * b1.clone() - a1.clone() * b3.clone());
        data.push(a1.clone() * b2.clone() - a2.clone() * b1.clone());

        ConstMatrix { data }
    }
}

// Cross product for 1x3 row vectors
impl<T> ConstMatrix<T, 1, 3>
where
    T: MatrixElement + std::fmt::Debug + Mul<Output = T> + Sub<Output = T>,
{
    /// Cross product of two 3D row vectors.
    ///
    /// Computes the vector cross product a × b for row vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let a: ConstMatrix<i32, 1, 3> = ConstMatrix::new(vec![1, 0, 0]);
    /// let b: ConstMatrix<i32, 1, 3> = ConstMatrix::new(vec![0, 1, 0]);
    /// let c = a.cross(&b);
    /// assert_eq!(c.get(0, 0), 0);
    /// assert_eq!(c.get(0, 1), 0);
    /// assert_eq!(c.get(0, 2), 1);
    /// ```
    pub fn cross(&self, other: &ConstMatrix<T, 1, 3>) -> ConstMatrix<T, 1, 3> {
        let a1 = &self.data[0];
        let a2 = &self.data[1];
        let a3 = &self.data[2];

        let b1 = &other.data[0];
        let b2 = &other.data[1];
        let b3 = &other.data[2];

        let mut data = Vec::with_capacity(3);
        data.push(a2.clone() * b3.clone() - a3.clone() * b2.clone());
        data.push(a3.clone() * b1.clone() - a1.clone() * b3.clone());
        data.push(a1.clone() * b2.clone() - a2.clone() * b1.clone());

        ConstMatrix { data }
    }
}

// Approximate equality for floating-point matrices
impl<T, const R: usize, const C: usize> ConstMatrix<T, R, C>
where
    T: MatrixElement + std::fmt::Debug + super::ToleranceOps + Sub<Output = T> + Clone,
    T::Abs: PartialOrd,
{
    /// Check if two matrices are approximately equal within a tolerance.
    ///
    /// Returns true if all corresponding elements differ by at most `tol`.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let a: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0]);
    /// let b: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0001, 2.0001, 3.0001, 4.0001]);
    /// assert!(a.approx_eq(&b, 0.001));
    /// assert!(!a.approx_eq(&b, 0.00001));
    /// ```
    pub fn approx_eq(&self, other: &ConstMatrix<T, R, C>, tol: T::Abs) -> bool {
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(a, b)| (a.clone() - b.clone()).abs_val() <= tol)
    }
}

// ============================================================================
// Norms (available for all matrix sizes)
// ============================================================================

impl<T, const R: usize, const C: usize> ConstMatrix<T, R, C>
where
    T: MatrixElement + std::fmt::Debug + super::norm::Abs,
    T::Output: Add<Output = T::Output> + Mul<Output = T::Output> + PartialOrd + From<u8> + Clone,
{
    /// Frobenius norm: sqrt(sum of squares of all elements).
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![3.0, 4.0, 0.0, 0.0]);
    /// assert_eq!(m.frobenius_norm(), 5.0); // sqrt(9 + 16) = 5
    /// ```
    pub fn frobenius_norm(&self) -> T::Output
    where
        T::Output: super::norm::Sqrt,
    {
        let sum = self
            .data
            .iter()
            .map(|x| {
                let abs_val = x.clone().abs();
                abs_val.clone() * abs_val.clone()
            })
            .fold(T::Output::from(0u8), |acc, x| acc + x);

        sum.sqrt()
    }

    /// Infinity norm: maximum absolute row sum.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, -2, 3, -4]);
    /// assert_eq!(m.inf_norm(), 7); // max(|1| + |-2|, |3| + |-4|) = max(3, 7) = 7
    /// ```
    pub fn inf_norm(&self) -> T::Output {
        (0..R)
            .map(|row| {
                self.row(row)
                    .iter()
                    .map(|x| x.clone().abs())
                    .fold(T::Output::from(0u8), |acc, x| acc + x)
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(T::Output::from(0u8))
    }

    /// One norm: maximum absolute column sum.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, -2, 3, -4]);
    /// assert_eq!(m.one_norm(), 6); // max(|1| + |3|, |-2| + |-4|) = max(4, 6) = 6
    /// ```
    pub fn one_norm(&self) -> T::Output {
        (0..C)
            .map(|col| {
                self.col_iter(col)
                    .map(|x| x.clone().abs())
                    .fold(T::Output::from(0u8), |acc, x| acc + x)
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(T::Output::from(0u8))
    }
}

// ============================================================================
// 2x2 Matrix Operations
// ============================================================================

impl<T> ConstMatrix<T, 2, 2>
where
    T: MatrixElement + std::fmt::Debug,
{
    /// Compute the determinant of a 2x2 matrix using the closed-form formula.
    ///
    /// For a 2x2 matrix [[a, b], [c, d]], the determinant is ad - bc.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    /// assert_eq!(m.determinant(), -2); // 1*4 - 2*3 = -2
    /// ```
    pub fn determinant(&self) -> T
    where
        T: Mul<Output = T> + Sub<Output = T>,
    {
        // [[a, b], [c, d]] -> ad - bc
        let a = &self.data[0];
        let b = &self.data[1];
        let c = &self.data[2];
        let d = &self.data[3];

        a.clone() * d.clone() - b.clone() * c.clone()
    }

    /// Compute the inverse of a 2x2 matrix using the closed-form formula.
    ///
    /// Returns `None` if the matrix is singular (determinant is zero for integers,
    /// or near-zero for floating-point types).
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![4.0, 7.0, 2.0, 6.0]);
    /// let inv = m.inverse().unwrap();
    /// assert!((inv.get(0, 0) - 0.6).abs() < 1e-10);
    /// ```
    pub fn inverse(&self) -> Option<ConstMatrix<T, 2, 2>>
    where
        T: Mul<Output = T> + Sub<Output = T> + Div<Output = T> + Neg<Output = T> + super::ToleranceOps,
        T::Abs: MatrixElement + Add<Output = T::Abs> + Mul<Output = T::Abs> + PartialOrd,
    {
        let det = self.determinant();

        // Check if determinant is (near) zero
        let tolerance = scaled_tolerance::<T, 2>(&self.data);
        if det.abs_val() <= tolerance {
            return None;
        }

        // [[a, b], [c, d]]^-1 = (1/det) * [[d, -b], [-c, a]]
        let a = &self.data[0];
        let b = &self.data[1];
        let c = &self.data[2];
        let d = &self.data[3];

        let mut data = Vec::with_capacity(4);
        data.push(d.clone() / det.clone());
        data.push(-b.clone() / det.clone());
        data.push(-c.clone() / det.clone());
        data.push(a.clone() / det.clone());

        Some(ConstMatrix { data })
    }
}

// ============================================================================
// 3x3 Matrix Operations
// ============================================================================

impl<T> ConstMatrix<T, 3, 3>
where
    T: MatrixElement + std::fmt::Debug,
{
    /// Compute the determinant of a 3x3 matrix using the closed-form formula.
    ///
    /// Uses the rule of Sarrus (diagonal products).
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<i32, 3, 3> = ConstMatrix::new(vec![
    ///     1, 2, 3,
    ///     0, 1, 4,
    ///     5, 6, 0,
    /// ]);
    /// assert_eq!(m.determinant(), 1); // 1*(1*0-4*6) - 2*(0*0-4*5) + 3*(0*6-1*5) = 1
    /// ```
    pub fn determinant(&self) -> T
    where
        T: Mul<Output = T> + Add<Output = T> + Sub<Output = T>,
    {
        // Row-major: [0,1,2, 3,4,5, 6,7,8]
        // [[a, b, c], [d, e, f], [g, h, i]]
        let a = &self.data[0];
        let b = &self.data[1];
        let c = &self.data[2];
        let d = &self.data[3];
        let e = &self.data[4];
        let f = &self.data[5];
        let g = &self.data[6];
        let h = &self.data[7];
        let i = &self.data[8];

        // det = a(ei - fh) - b(di - fg) + c(dh - eg)
        let term1 = a.clone() * (e.clone() * i.clone() - f.clone() * h.clone());
        let term2 = b.clone() * (d.clone() * i.clone() - f.clone() * g.clone());
        let term3 = c.clone() * (d.clone() * h.clone() - e.clone() * g.clone());

        term1 - term2 + term3
    }

    /// Compute the inverse of a 3x3 matrix using the adjugate matrix formula.
    ///
    /// Returns `None` if the matrix is singular.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<f64, 3, 3> = ConstMatrix::new(vec![
    ///     1.0, 2.0, 3.0,
    ///     0.0, 1.0, 4.0,
    ///     5.0, 6.0, 0.0,
    /// ]);
    /// let inv = m.inverse().unwrap();
    /// // Verify: M * M^-1 ≈ I
    /// let identity = &m * &inv;
    /// assert!(identity.approx_eq(&ConstMatrix::identity(), 1e-10));
    /// ```
    pub fn inverse(&self) -> Option<ConstMatrix<T, 3, 3>>
    where
        T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Neg<Output = T> + super::ToleranceOps,
        T::Abs: MatrixElement + Add<Output = T::Abs> + Mul<Output = T::Abs> + PartialOrd,
    {
        let det = self.determinant();

        // Check if determinant is (near) zero
        let tolerance = scaled_tolerance::<T, 3>(&self.data);
        if det.abs_val() <= tolerance {
            return None;
        }

        // Compute cofactor matrix (adjugate is transpose of cofactor matrix)
        let a = &self.data[0];
        let b = &self.data[1];
        let c = &self.data[2];
        let d = &self.data[3];
        let e = &self.data[4];
        let f = &self.data[5];
        let g = &self.data[6];
        let h = &self.data[7];
        let i = &self.data[8];

        // Cofactor matrix elements
        let c00 = e.clone() * i.clone() - f.clone() * h.clone();
        let c01 = -(d.clone() * i.clone() - f.clone() * g.clone());
        let c02 = d.clone() * h.clone() - e.clone() * g.clone();

        let c10 = -(b.clone() * i.clone() - c.clone() * h.clone());
        let c11 = a.clone() * i.clone() - c.clone() * g.clone();
        let c12 = -(a.clone() * h.clone() - b.clone() * g.clone());

        let c20 = b.clone() * f.clone() - c.clone() * e.clone();
        let c21 = -(a.clone() * f.clone() - c.clone() * d.clone());
        let c22 = a.clone() * e.clone() - b.clone() * d.clone();

        // Adjugate is transpose of cofactor, divided by determinant
        let mut data = Vec::with_capacity(9);
        data.push(c00 / det.clone());
        data.push(c10 / det.clone());
        data.push(c20 / det.clone());
        data.push(c01 / det.clone());
        data.push(c11 / det.clone());
        data.push(c21 / det.clone());
        data.push(c02 / det.clone());
        data.push(c12 / det.clone());
        data.push(c22 / det.clone());

        Some(ConstMatrix { data })
    }
}

// ============================================================================
// 4x4 Matrix Operations
// ============================================================================

impl<T> ConstMatrix<T, 4, 4>
where
    T: MatrixElement + std::fmt::Debug,
{
    /// Compute the determinant of a 4x4 matrix using cofactor expansion.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<i32, 4, 4> = ConstMatrix::identity();
    /// assert_eq!(m.determinant(), 1);
    /// ```
    pub fn determinant(&self) -> T
    where
        T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Neg<Output = T>,
    {
        // Cofactor expansion along first row
        let mut det = T::zero();

        for j in 0..4 {
            let cofactor = self.cofactor_3x3(0, j);
            let sign = if j % 2 == 0 { T::one() } else { -T::one() };
            det = det + (sign * self.data[j].clone() * cofactor);
        }

        det
    }

    /// Helper: compute 3x3 determinant of submatrix (excluding row i, col j).
    fn cofactor_3x3(&self, exclude_row: usize, exclude_col: usize) -> T
    where
        T: Mul<Output = T> + Add<Output = T> + Sub<Output = T>,
    {
        // Extract 3x3 submatrix
        let mut sub = Vec::with_capacity(9);
        for row in 0..4 {
            if row == exclude_row {
                continue;
            }
            for col in 0..4 {
                if col == exclude_col {
                    continue;
                }
                sub.push(self.data[row * 4 + col].clone());
            }
        }

        // Compute 3x3 determinant manually
        let a = &sub[0];
        let b = &sub[1];
        let c = &sub[2];
        let d = &sub[3];
        let e = &sub[4];
        let f = &sub[5];
        let g = &sub[6];
        let h = &sub[7];
        let i = &sub[8];

        let term1 = a.clone() * (e.clone() * i.clone() - f.clone() * h.clone());
        let term2 = b.clone() * (d.clone() * i.clone() - f.clone() * g.clone());
        let term3 = c.clone() * (d.clone() * h.clone() - e.clone() * g.clone());

        term1 - term2 + term3
    }

    /// Compute the inverse of a 4x4 matrix.
    ///
    /// Uses the adjugate matrix method with cofactor expansion.
    /// Returns `None` if the matrix is singular.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<f64, 4, 4> = ConstMatrix::identity();
    /// let inv = m.inverse().unwrap();
    /// assert!(inv.approx_eq(&ConstMatrix::identity(), 1e-10));
    /// ```
    pub fn inverse(&self) -> Option<ConstMatrix<T, 4, 4>>
    where
        T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Neg<Output = T> + super::ToleranceOps,
        T::Abs: MatrixElement + Add<Output = T::Abs> + Mul<Output = T::Abs> + PartialOrd,
    {
        let det = self.determinant();

        // Check if determinant is (near) zero
        let tolerance = scaled_tolerance::<T, 4>(&self.data);
        if det.abs_val() <= tolerance {
            return None;
        }

        // Compute adjugate matrix (transpose of cofactor matrix)
        let mut data = Vec::with_capacity(16);

        for j in 0..4 {
            for i in 0..4 {
                let cofactor = self.cofactor_3x3(i, j);
                let sign = if (i + j) % 2 == 0 { T::one() } else { -T::one() };
                data.push((sign * cofactor) / det.clone());
            }
        }

        Some(ConstMatrix { data })
    }
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

// Addition: ConstMatrix + ConstMatrix (owned + owned)
// Dimensions must match exactly - enforced by type system!
impl<T, const R: usize, const C: usize> Add for ConstMatrix<T, R, C>
where
    T: MatrixElement + std::fmt::Debug + Add<Output = T>,
{
    type Output = ConstMatrix<T, R, C>;

    fn add(mut self, other: ConstMatrix<T, R, C>) -> Self::Output {
        // No dimension checks needed - type system guarantees it!
        for (lhs, rhs) in self.data.iter_mut().zip(other.data.into_iter()) {
            *lhs = lhs.clone() + rhs;
        }
        self
    }
}

// Addition: ConstMatrix + &ConstMatrix (owned + ref)
impl<T, const R: usize, const C: usize> Add<&ConstMatrix<T, R, C>> for ConstMatrix<T, R, C>
where
    T: MatrixElement + std::fmt::Debug + Add<Output = T>,
{
    type Output = ConstMatrix<T, R, C>;

    fn add(mut self, other: &ConstMatrix<T, R, C>) -> Self::Output {
        for (lhs, rhs) in self.data.iter_mut().zip(other.data.iter()) {
            *lhs = lhs.clone() + rhs.clone();
        }
        self
    }
}

// Addition: &ConstMatrix + &ConstMatrix (ref + ref)
impl<T, const R: usize, const C: usize> Add for &ConstMatrix<T, R, C>
where
    T: MatrixElement + std::fmt::Debug + Add<Output = T>,
{
    type Output = ConstMatrix<T, R, C>;

    fn add(self, other: &ConstMatrix<T, R, C>) -> Self::Output {
        let mut data = Vec::with_capacity(R * C);
        for i in 0..self.data.len() {
            data.push(self.data[i].clone() + other.data[i].clone());
        }
        ConstMatrix { data }
    }
}

// Subtraction: ConstMatrix - ConstMatrix (owned - owned)
impl<T, const R: usize, const C: usize> Sub for ConstMatrix<T, R, C>
where
    T: MatrixElement + std::fmt::Debug + Sub<Output = T>,
{
    type Output = ConstMatrix<T, R, C>;

    fn sub(mut self, other: ConstMatrix<T, R, C>) -> Self::Output {
        for (lhs, rhs) in self.data.iter_mut().zip(other.data.into_iter()) {
            *lhs = lhs.clone() - rhs;
        }
        self
    }
}

// Subtraction: ConstMatrix - &ConstMatrix (owned - ref)
impl<T, const R: usize, const C: usize> Sub<&ConstMatrix<T, R, C>> for ConstMatrix<T, R, C>
where
    T: MatrixElement + std::fmt::Debug + Sub<Output = T>,
{
    type Output = ConstMatrix<T, R, C>;

    fn sub(mut self, other: &ConstMatrix<T, R, C>) -> Self::Output {
        for (lhs, rhs) in self.data.iter_mut().zip(other.data.iter()) {
            *lhs = lhs.clone() - rhs.clone();
        }
        self
    }
}

// Subtraction: &ConstMatrix - &ConstMatrix (ref - ref)
impl<T, const R: usize, const C: usize> Sub for &ConstMatrix<T, R, C>
where
    T: MatrixElement + std::fmt::Debug + Sub<Output = T>,
{
    type Output = ConstMatrix<T, R, C>;

    fn sub(self, other: &ConstMatrix<T, R, C>) -> Self::Output {
        let mut data = Vec::with_capacity(R * C);
        for i in 0..self.data.len() {
            data.push(self.data[i].clone() - other.data[i].clone());
        }
        ConstMatrix { data }
    }
}

// Matrix multiplication: ConstMatrix<R, K> * ConstMatrix<K, C> -> ConstMatrix<R, C>
// Inner dimension K must match - enforced by type system!
impl<T, const R: usize, const K: usize, const C: usize> Mul<ConstMatrix<T, K, C>>
    for ConstMatrix<T, R, K>
where
    T: MatrixElement + std::fmt::Debug + Mul<Output = T> + Add<Output = T>,
{
    type Output = ConstMatrix<T, R, C>;

    fn mul(self, other: ConstMatrix<T, K, C>) -> Self::Output {
        // Transpose RHS for cache locality (same optimization as dynamic Matrix)
        let other_t = other.transpose();

        let mut data = Vec::with_capacity(R * C);

        for i in 0..R {
            let row = &self.data[i * K..(i + 1) * K];
            for j in 0..C {
                let col = &other_t.data[j * K..(j + 1) * K];
                let mut sum = T::zero();
                for k in 0..K {
                    sum = sum + (row[k].clone() * col[k].clone());
                }
                data.push(sum);
            }
        }

        ConstMatrix { data }
    }
}

// Matrix multiplication: ConstMatrix<R, K> * &ConstMatrix<K, C> -> ConstMatrix<R, C>
impl<T, const R: usize, const K: usize, const C: usize> Mul<&ConstMatrix<T, K, C>>
    for ConstMatrix<T, R, K>
where
    T: MatrixElement + std::fmt::Debug + Mul<Output = T> + Add<Output = T>,
{
    type Output = ConstMatrix<T, R, C>;

    fn mul(self, other: &ConstMatrix<T, K, C>) -> Self::Output {
        let other_t = other.transpose();

        let mut data = Vec::with_capacity(R * C);

        for i in 0..R {
            let row = &self.data[i * K..(i + 1) * K];
            for j in 0..C {
                let col = &other_t.data[j * K..(j + 1) * K];
                let mut sum = T::zero();
                for k in 0..K {
                    sum = sum + (row[k].clone() * col[k].clone());
                }
                data.push(sum);
            }
        }

        ConstMatrix { data }
    }
}

// Matrix multiplication: &ConstMatrix<R, K> * &ConstMatrix<K, C> -> ConstMatrix<R, C>
impl<T, const R: usize, const K: usize, const C: usize> Mul<&ConstMatrix<T, K, C>>
    for &ConstMatrix<T, R, K>
where
    T: MatrixElement + std::fmt::Debug + Mul<Output = T> + Add<Output = T>,
{
    type Output = ConstMatrix<T, R, C>;

    fn mul(self, other: &ConstMatrix<T, K, C>) -> Self::Output {
        let other_t = other.transpose();

        let mut data = Vec::with_capacity(R * C);

        for i in 0..R {
            let row = &self.data[i * K..(i + 1) * K];
            for j in 0..C {
                let col = &other_t.data[j * K..(j + 1) * K];
                let mut sum = T::zero();
                for k in 0..K {
                    sum = sum + (row[k].clone() * col[k].clone());
                }
                data.push(sum);
            }
        }

        ConstMatrix { data }
    }
}

// Scalar multiplication: ConstMatrix * scalar
impl<T, const R: usize, const C: usize> Mul<T> for ConstMatrix<T, R, C>
where
    T: MatrixElement + std::fmt::Debug + Mul<Output = T>,
{
    type Output = ConstMatrix<T, R, C>;

    fn mul(mut self, scalar: T) -> Self::Output {
        for elem in self.data.iter_mut() {
            *elem = elem.clone() * scalar.clone();
        }
        self
    }
}

// Scalar multiplication: &ConstMatrix * scalar
impl<T, const R: usize, const C: usize> Mul<T> for &ConstMatrix<T, R, C>
where
    T: MatrixElement + std::fmt::Debug + Mul<Output = T>,
{
    type Output = ConstMatrix<T, R, C>;

    fn mul(self, scalar: T) -> Self::Output {
        let mut data = Vec::with_capacity(R * C);
        for i in 0..self.data.len() {
            data.push(self.data[i].clone() * scalar.clone());
        }
        ConstMatrix { data }
    }
}

// Scalar multiplication: scalar * ConstMatrix (f64)
impl<const R: usize, const C: usize> Mul<ConstMatrix<f64, R, C>> for f64 {
    type Output = ConstMatrix<f64, R, C>;

    fn mul(self, mut matrix: ConstMatrix<f64, R, C>) -> Self::Output {
        for elem in matrix.data.iter_mut() {
            *elem = self * *elem;
        }
        matrix
    }
}

// Scalar multiplication: scalar * &ConstMatrix (f64)
impl<const R: usize, const C: usize> Mul<&ConstMatrix<f64, R, C>> for f64 {
    type Output = ConstMatrix<f64, R, C>;

    fn mul(self, matrix: &ConstMatrix<f64, R, C>) -> Self::Output {
        let mut data = Vec::with_capacity(R * C);
        for i in 0..matrix.data.len() {
            data.push(self * matrix.data[i]);
        }
        ConstMatrix { data }
    }
}

// Scalar multiplication: scalar * ConstMatrix (f32)
impl<const R: usize, const C: usize> Mul<ConstMatrix<f32, R, C>> for f32 {
    type Output = ConstMatrix<f32, R, C>;

    fn mul(self, mut matrix: ConstMatrix<f32, R, C>) -> Self::Output {
        for elem in matrix.data.iter_mut() {
            *elem = self * *elem;
        }
        matrix
    }
}

// Scalar multiplication: scalar * &ConstMatrix (f32)
impl<const R: usize, const C: usize> Mul<&ConstMatrix<f32, R, C>> for f32 {
    type Output = ConstMatrix<f32, R, C>;

    fn mul(self, matrix: &ConstMatrix<f32, R, C>) -> Self::Output {
        let mut data = Vec::with_capacity(R * C);
        for i in 0..matrix.data.len() {
            data.push(self * matrix.data[i]);
        }
        ConstMatrix { data }
    }
}

// Scalar multiplication: scalar * ConstMatrix (i32)
impl<const R: usize, const C: usize> Mul<ConstMatrix<i32, R, C>> for i32 {
    type Output = ConstMatrix<i32, R, C>;

    fn mul(self, mut matrix: ConstMatrix<i32, R, C>) -> Self::Output {
        for elem in matrix.data.iter_mut() {
            *elem = self * *elem;
        }
        matrix
    }
}

// Scalar multiplication: scalar * &ConstMatrix (i32)
impl<const R: usize, const C: usize> Mul<&ConstMatrix<i32, R, C>> for i32 {
    type Output = ConstMatrix<i32, R, C>;

    fn mul(self, matrix: &ConstMatrix<i32, R, C>) -> Self::Output {
        let mut data = Vec::with_capacity(R * C);
        for i in 0..matrix.data.len() {
            data.push(self * matrix.data[i]);
        }
        ConstMatrix { data }
    }
}

// ============================================================================
// Conversions
// ============================================================================

// Conversion from ConstMatrix to dynamic Matrix (always succeeds)
impl<T, const R: usize, const C: usize> From<ConstMatrix<T, R, C>> for Matrix<T>
where
    T: MatrixElement + std::fmt::Debug,
{
    fn from(cm: ConstMatrix<T, R, C>) -> Self {
        Matrix::new(R, C, cm.data)
    }
}

// Conversion from dynamic Matrix to ConstMatrix (fallible - dimensions must match)
impl<T, const R: usize, const C: usize> TryFrom<Matrix<T>> for ConstMatrix<T, R, C>
where
    T: MatrixElement + std::fmt::Debug,
{
    type Error = MatrixError;

    fn try_from(m: Matrix<T>) -> Result<Self, Self::Error> {
        if m.rows() != R || m.cols() != C {
            return Err(MatrixError::DimensionMismatch);
        }

        // Extract data using into_data helper (we'll need to add this to Matrix)
        // For now, we need to work around the private fields
        // We'll create a new ConstMatrix by cloning the data
        let mut data = Vec::with_capacity(R * C);
        for row in 0..R {
            for col in 0..C {
                data.push(m.get(row, col));
            }
        }

        Ok(ConstMatrix { data })
    }
}
