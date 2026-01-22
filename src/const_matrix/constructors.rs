//! ConstMatrix constructors.

use crate::matrix::Matrix;
use crate::traits::MatrixElement;
use std::ops::{Add, Mul};

use super::ConstMatrix;

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
