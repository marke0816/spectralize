//! Conversions between ConstMatrix and Matrix.

use crate::error::MatrixError;
use crate::matrix::Matrix;
use crate::traits::MatrixElement;

use super::ConstMatrix;

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
