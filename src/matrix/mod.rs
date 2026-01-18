pub mod append;
pub mod arithmetic;
pub mod const_matrix;
pub mod decomposition;
pub mod element;
pub mod norm;

use crate::matrix::decomposition::PivotOrd;
use crate::matrix::norm::Abs;
pub use const_matrix::ConstMatrix;
pub use element::{MatrixElement, NanCheck, ToleranceOps};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixError {
    DimensionMismatch,
    IndexOutOfBounds,
    PermutationDuplicateIndex,
    PermutationIndexOutOfBounds,
    PermutationLengthMismatch,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T: MatrixElement + std::fmt::Debug> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

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

    /// Get the number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Zero-cost reference getter
    pub fn get_ref(&self, row: usize, col: usize) -> &T {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        &self.data[row * self.cols + col]
    }

    /// Checked reference getter
    pub fn try_get_ref(&self, row: usize, col: usize) -> Result<&T, MatrixError> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixError::IndexOutOfBounds);
        }
        Ok(&self.data[row * self.cols + col])
    }

    /// Safe getter that returns a clone of the element
    pub fn get(&self, row: usize, col: usize) -> T {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        self.data[row * self.cols + col].clone()
    }

    /// Checked getter that returns a clone of the element
    pub fn try_get(&self, row: usize, col: usize) -> Result<T, MatrixError> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixError::IndexOutOfBounds);
        }
        Ok(self.data[row * self.cols + col].clone())
    }

    /// Safe setter
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        self.data[row * self.cols + col] = value;
    }

    /// Checked setter
    pub fn try_set(&mut self, row: usize, col: usize, value: T) -> Result<(), MatrixError> {
        if row >= self.rows || col >= self.cols {
            return Err(MatrixError::IndexOutOfBounds);
        }
        self.data[row * self.cols + col] = value;
        Ok(())
    }

    /// Borrowed slice over a row
    pub fn row(&self, row: usize) -> &[T] {
        assert!(row < self.rows, "Row index out of bounds");
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Checked row access
    pub fn try_row(&self, row: usize) -> Result<&[T], MatrixError> {
        if row >= self.rows {
            return Err(MatrixError::IndexOutOfBounds);
        }
        let start = row * self.cols;
        Ok(&self.data[start..start + self.cols])
    }

    /// Iterator over a column
    #[deprecated(note = "col() allocates; use col_iter() to avoid allocation when possible")]
    pub fn col(&self, col: usize) -> Vec<T> {
        assert!(col < self.cols, "Column index out of bounds");
        (0..self.rows)
            .map(|r| self.data[r * self.cols + col].clone())
            .collect()
    }

    /// Checked column access
    pub fn try_col(&self, col: usize) -> Result<Vec<T>, MatrixError> {
        if col >= self.cols {
            return Err(MatrixError::IndexOutOfBounds);
        }
        Ok((0..self.rows)
            .map(|r| self.data[r * self.cols + col].clone())
            .collect())
    }

    /// Iterator over a column without allocation
    pub fn col_iter(&self, col: usize) -> impl Iterator<Item = &T> {
        assert!(col < self.cols, "Column index out of bounds");
        (0..self.rows).map(move |r| &self.data[r * self.cols + col])
    }

    /// Compute the transpose of the matrix
    pub fn transpose(&self) -> Self {
        // Pre-allocate for performance
        let mut data = Vec::with_capacity(self.rows * self.cols);

        // Iterate column-major to populate transposed row-major layout
        for col in 0..self.cols {
            for row in 0..self.rows {
                data.push(self.data[row * self.cols + col].clone());
            }
        }

        Self {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    /// Compute the trace (sum of diagonal elements) of a square matrix
    pub fn trace(&self) -> T
    where
        T: std::ops::Add<Output = T>,
    {
        assert_eq!(self.rows, self.cols, "Matrix must be square for trace");

        // Efficiently sum diagonal elements without intermediate allocations
        (0..self.rows)
            .map(|i| self.data[i * self.cols + i].clone())
            .fold(T::zero(), |acc, x| acc + x)
    }
}

impl<T> Matrix<T>
where
    T: MatrixElement + std::fmt::Debug,
{
    pub fn try_with_cols(&self, other: &Matrix<T>) -> Result<Self, MatrixError> {
        if self.rows != other.rows {
            return Err(MatrixError::DimensionMismatch);
        }
        Ok(self.with_cols(other))
    }

    pub fn try_with_rows(&self, other: &Matrix<T>) -> Result<Self, MatrixError> {
        if self.cols != other.cols {
            return Err(MatrixError::DimensionMismatch);
        }
        Ok(self.with_rows(other))
    }

    pub fn try_with_row_vec(&self, row: &[T]) -> Result<Self, MatrixError> {
        if self.cols != row.len() {
            return Err(MatrixError::DimensionMismatch);
        }
        Ok(self.with_row_vec(row))
    }

    pub fn try_with_col_vec(&self, col: &[T]) -> Result<Self, MatrixError> {
        if self.rows != col.len() {
            return Err(MatrixError::DimensionMismatch);
        }
        Ok(self.with_col_vec(col))
    }
}

impl<T> Matrix<T>
where
    T: MatrixElement + std::fmt::Debug + Add<Output = T>,
{
    pub fn try_add(&self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::DimensionMismatch);
        }
        Ok(self + other)
    }
}

impl<T> Matrix<T>
where
    T: MatrixElement + std::fmt::Debug + Sub<Output = T>,
{
    pub fn try_sub(&self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MatrixError::DimensionMismatch);
        }
        Ok(self - other)
    }
}

impl<T> Matrix<T>
where
    T: MatrixElement + std::fmt::Debug + Mul<Output = T> + Add<Output = T>,
{
    pub fn try_mul(&self, other: &Matrix<T>) -> Result<Matrix<T>, MatrixError> {
        if self.cols != other.rows {
            return Err(MatrixError::DimensionMismatch);
        }
        Ok(self * other)
    }
}

impl<T> Matrix<T>
where
    T: MatrixElement + std::fmt::Debug + Add<Output = T>,
{
    pub fn try_trace(&self) -> Result<T, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }
        Ok(self.trace())
    }
}

impl<T> Matrix<T>
where
    T: MatrixElement
        + std::fmt::Debug
        + ToleranceOps
        + PivotOrd
        + Div<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + Neg<Output = T>
        + Clone
        + Abs<Output = T::Abs>,
    T::Abs: MatrixElement + Add<Output = T::Abs> + Mul<Output = T::Abs> + PartialOrd + NanCheck,
{
    pub fn try_determinant(&self) -> Result<T, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }
        Ok(self.determinant())
    }

    pub fn try_determinant_with_tol(&self, tolerance: T::Abs) -> Result<T, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }
        Ok(self.determinant_with_tol(tolerance))
    }

    pub fn try_is_invertible(&self) -> Result<bool, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }
        Ok(self.is_invertible())
    }

    pub fn try_is_invertible_with_tol(&self, tolerance: T::Abs) -> Result<bool, MatrixError> {
        if self.rows != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }
        Ok(self.is_invertible_with_tol(tolerance))
    }
}

impl<T> Matrix<T>
where
    T: MatrixElement + std::fmt::Debug + ToleranceOps + Sub<Output = T> + Clone,
    T::Abs: PartialOrd,
{
    pub fn approx_eq(&self, other: &Matrix<T>, tol: T::Abs) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }

        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(a, b)| (a.clone() - b.clone()).abs_val() <= tol)
    }
}

#[cfg(test)]
mod const_matrix_tests;
#[cfg(test)]
mod tests;
