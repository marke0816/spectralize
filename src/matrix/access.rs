//! Matrix accessors and basic operations (get, set, transpose, trace, etc.).

use crate::error::MatrixError;
use crate::traits::{MatrixElement, ToleranceOps};

use super::Matrix;

impl<T: MatrixElement + std::fmt::Debug> Matrix<T> {
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
    T: MatrixElement + std::fmt::Debug + ToleranceOps + std::ops::Sub<Output = T> + Clone,
    T::Abs: PartialOrd,
{
    /// Check if two matrices are approximately equal within a given tolerance.
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
