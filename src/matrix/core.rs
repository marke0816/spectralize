//! Core Matrix type definition and wrapper methods for checked operations.

use crate::error::MatrixError;
use crate::traits::{Abs, MatrixElement, NanCheck, PivotOrd, ToleranceOps};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A dynamically-sized matrix with row-major storage.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T: MatrixElement + std::fmt::Debug> {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) data: Vec<T>,
}

// Checked wrappers for concatenation operations
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

// Checked wrappers for arithmetic operations
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

// Checked wrappers for linear algebra operations
impl<T> Matrix<T>
where
    T: MatrixElement
        + std::fmt::Debug
        + ToleranceOps
        + PivotOrd
        + Div<Output = T>
        + Mul<Output = T>
        + Add<Output = T>
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
