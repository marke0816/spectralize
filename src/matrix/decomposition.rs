use super::{Matrix, MatrixElement};
use std::ops::{Div, Mul, Neg, Sub};

/// Internal helper struct that stores the result of PLU decomposition.
/// The L and U matrices are stored in a single matrix:
/// - Lower triangle (below diagonal) contains L with implicit 1s on diagonal
/// - Upper triangle (including diagonal) contains U
/// This packed storage is standard in numerical linear algebra for efficiency.
#[derive(Debug, Clone)]
struct PLUDecomposition<T: MatrixElement + std::fmt::Debug> {
    /// Combined L and U matrices in packed form
    lu: Matrix<T>,
    /// Number of row swaps performed (determines sign of determinant)
    row_swaps: usize,
    /// Whether the matrix is singular (has a zero pivot)
    is_singular: bool,
}

/// Trait for types that support magnitude comparison for pivoting.
/// This is needed to find the largest absolute value element for partial pivoting.
pub trait PivotOrd: PartialOrd + Sized {
    /// Return a value suitable for pivot comparison (e.g., absolute value for signed types)
    fn pivot_key(&self) -> Self;
}

// For unsigned integers, the pivot key is the value itself
impl PivotOrd for u8 {
    fn pivot_key(&self) -> Self {
        *self
    }
}

impl PivotOrd for u16 {
    fn pivot_key(&self) -> Self {
        *self
    }
}

impl PivotOrd for u32 {
    fn pivot_key(&self) -> Self {
        *self
    }
}

impl PivotOrd for u64 {
    fn pivot_key(&self) -> Self {
        *self
    }
}

impl PivotOrd for u128 {
    fn pivot_key(&self) -> Self {
        *self
    }
}

impl PivotOrd for usize {
    fn pivot_key(&self) -> Self {
        *self
    }
}

// For signed integers, the pivot key is the absolute value
impl PivotOrd for i8 {
    fn pivot_key(&self) -> Self {
        self.abs()
    }
}

impl PivotOrd for i16 {
    fn pivot_key(&self) -> Self {
        self.abs()
    }
}

impl PivotOrd for i32 {
    fn pivot_key(&self) -> Self {
        self.abs()
    }
}

impl PivotOrd for i64 {
    fn pivot_key(&self) -> Self {
        self.abs()
    }
}

impl PivotOrd for i128 {
    fn pivot_key(&self) -> Self {
        self.abs()
    }
}

impl PivotOrd for isize {
    fn pivot_key(&self) -> Self {
        self.abs()
    }
}

// For floating point, use absolute value
impl PivotOrd for f32 {
    fn pivot_key(&self) -> Self {
        self.abs()
    }
}

impl PivotOrd for f64 {
    fn pivot_key(&self) -> Self {
        self.abs()
    }
}

impl<T> PLUDecomposition<T>
where
    T: MatrixElement + std::fmt::Debug + PivotOrd + Div<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    /// Compute PLU decomposition with partial pivoting for a square matrix.
    ///
    /// This implements Gaussian elimination with row pivoting:
    /// 1. For each column k, find the row with largest absolute value (partial pivoting)
    /// 2. Swap that row with row k (if different)
    /// 3. Eliminate all entries below the pivot
    /// 4. Store multipliers in the lower triangle
    ///
    /// Returns a PLUDecomposition storing:
    /// - Combined L/U matrix (L below diagonal with implicit 1s, U on/above diagonal)
    /// - Number of row swaps (for determinant sign: (-1)^swaps)
    /// - Singularity flag (true if any pivot is zero)
    fn decompose(matrix: &Matrix<T>) -> Self {
        assert_eq!(
            matrix.rows(),
            matrix.cols(),
            "PLU decomposition requires a square matrix"
        );

        let n = matrix.rows();

        // Clone the matrix for in-place elimination (optimal for performance)
        let mut lu = matrix.clone();
        let mut row_swaps = 0;
        let mut is_singular = false;

        // Gaussian elimination with partial pivoting
        for k in 0..n {
            // PARTIAL PIVOTING: Find the row with the largest absolute value in column k
            // This improves numerical stability and ensures we don't divide by small numbers
            let mut pivot_row = k;
            let mut max_pivot_key = lu.get(k, k).pivot_key();

            for i in (k + 1)..n {
                let candidate_key = lu.get(i, k).pivot_key();
                if candidate_key > max_pivot_key {
                    max_pivot_key = candidate_key;
                    pivot_row = i;
                }
            }

            // Check for zero pivot (indicates singular matrix)
            if lu.get(pivot_row, k) == T::zero() {
                is_singular = true;
                // Continue to check remaining pivots for complete decomposition
                continue;
            }

            // ROW SWAP: Exchange rows k and pivot_row if they differ
            // Each swap changes the sign of the determinant
            if pivot_row != k {
                for j in 0..n {
                    let temp = lu.get(k, j);
                    lu.set(k, j, lu.get(pivot_row, j));
                    lu.set(pivot_row, j, temp);
                }
                row_swaps += 1;
            }

            // ELIMINATION: Zero out all entries below the pivot in column k
            let pivot = lu.get(k, k);

            for i in (k + 1)..n {
                // Compute multiplier: L[i,k] = A[i,k] / A[k,k]
                let multiplier = lu.get(i, k) / pivot.clone();

                // Store the multiplier in lower triangle (L matrix)
                // Note: diagonal of L is implicitly 1, so we don't store it
                lu.set(i, k, multiplier.clone());

                // Update row i: row[i] -= multiplier * row[k]
                // Only need to update entries at and right of the diagonal
                // (entries to the left are already processed or will store L)
                for j in (k + 1)..n {
                    let update = lu.get(i, j) - (multiplier.clone() * lu.get(k, j));
                    lu.set(i, j, update);
                }
            }
        }

        PLUDecomposition {
            lu,
            row_swaps,
            is_singular,
        }
    }

    /// Check if the original matrix is invertible.
    /// A matrix is invertible if and only if all pivots (diagonal of U) are non-zero.
    fn is_invertible(&self) -> bool {
        !self.is_singular
    }

    /// Compute the determinant of the original matrix.
    ///
    /// For a PLU decomposition: det(A) = det(P) * det(L) * det(U)
    /// - det(P) = (-1)^(number of row swaps)
    /// - det(L) = 1 (lower triangular with 1s on diagonal)
    /// - det(U) = product of diagonal elements
    ///
    /// Therefore: det(A) = (-1)^(row_swaps) * product(diagonal of U)
    fn determinant(&self) -> T
    where
        T: Neg<Output = T>,
    {
        // Singular matrix has determinant zero
        if self.is_singular {
            return T::zero();
        }

        let n = self.lu.rows();

        // Compute product of diagonal elements (these are the pivots from U)
        let mut product = T::one();
        for i in 0..n {
            product = product * self.lu.get(i, i);
        }

        // Apply sign based on number of row swaps: (-1)^row_swaps
        // Even number of swaps: positive determinant
        // Odd number of swaps: negative determinant
        if self.row_swaps % 2 == 1 {
            -product
        } else {
            product
        }
    }
}

// Public API methods on Matrix<T>
impl<T> Matrix<T>
where
    T: MatrixElement + std::fmt::Debug,
{
    /// Check if the matrix is invertible (non-singular).
    ///
    /// A matrix is invertible if and only if its determinant is non-zero,
    /// which is equivalent to all pivots in the PLU decomposition being non-zero.
    ///
    /// This method performs a single PLU decomposition internally.
    ///
    /// # Panics
    /// Panics if the matrix is not square.
    ///
    /// # Example
    /// ```
    /// use spectralize::matrix::Matrix;
    ///
    /// let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    /// assert!(m.is_invertible());
    ///
    /// let singular = Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
    /// assert!(!singular.is_invertible());
    /// ```
    pub fn is_invertible(&self) -> bool
    where
        T: PivotOrd + Div<Output = T> + Mul<Output = T> + Sub<Output = T>,
    {
        let decomp = PLUDecomposition::decompose(self);
        decomp.is_invertible()
    }

    /// Compute the determinant of the matrix.
    ///
    /// The determinant is computed using PLU decomposition:
    /// det(A) = (-1)^(row_swaps) * product(diagonal of U)
    ///
    /// For singular matrices, returns zero.
    ///
    /// This method performs a single PLU decomposition internally.
    ///
    /// # Panics
    /// Panics if the matrix is not square.
    ///
    /// # Example
    /// ```
    /// use spectralize::matrix::Matrix;
    ///
    /// let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(m.determinant(), -2.0);
    ///
    /// let singular = Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
    /// assert_eq!(singular.determinant(), 0.0);
    /// ```
    pub fn determinant(&self) -> T
    where
        T: PivotOrd + Div<Output = T> + Mul<Output = T> + Sub<Output = T> + Neg<Output = T>,
    {
        let decomp = PLUDecomposition::decompose(self);
        decomp.determinant()
    }
}
