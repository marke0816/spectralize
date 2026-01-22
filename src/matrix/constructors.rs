//! Matrix constructors.

use crate::error::MatrixError;
use crate::traits::MatrixElement;

use super::Matrix;

impl<T: MatrixElement + std::fmt::Debug> Matrix<T> {
    /// Create a new matrix from a Vec<T>
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(rows * cols, data.len());
        Self { rows, cols, data }
    }

    /// Create a rows x cols zero matrix
    pub fn zero(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![T::zero(); rows * cols],
        }
    }

    /// Create a rows x cols identity matrix
    pub fn identity(rows: usize, cols: usize) -> Self {
        let mut data = vec![T::zero(); rows * cols];

        (0..rows.min(cols)).for_each(|i| data[i * cols + i] = T::one());

        Self { rows, cols, data }
    }

    /// Create a permutation matrix from a vector of column indices.
    /// The `perm` vector should contain integers 1..=cols, one for each row
    pub fn perm(rows: usize, cols: usize, perm: Vec<usize>) -> Self {
        assert_eq!(
            rows,
            perm.len(),
            "Length of permutation vector must match rows"
        );
        let mut data = vec![T::zero(); rows * cols];
        let mut used = vec![false; cols];

        for (row, &col_index) in perm.iter().enumerate() {
            assert!(
                col_index >= 1 && col_index <= cols,
                "Column indices must be 1-based and <= cols"
            );
            let col = col_index - 1;
            assert!(!used[col], "Permutation vector contains duplicate indices");
            used[col] = true;
            data[row * cols + col] = T::one();
        }

        Self { rows, cols, data }
    }

    /// Checked version of `perm` that returns an error instead of panicking.
    pub fn try_perm(rows: usize, cols: usize, perm: Vec<usize>) -> Result<Self, MatrixError> {
        if rows != perm.len() {
            return Err(MatrixError::PermutationLengthMismatch);
        }
        let mut data = vec![T::zero(); rows * cols];
        let mut used = vec![false; cols];

        for (row, &col_index) in perm.iter().enumerate() {
            if col_index == 0 || col_index > cols {
                return Err(MatrixError::PermutationIndexOutOfBounds);
            }
            let col = col_index - 1;
            if used[col] {
                return Err(MatrixError::PermutationDuplicateIndex);
            }
            used[col] = true;
            data[row * cols + col] = T::one();
        }

        Ok(Self { rows, cols, data })
    }
}
