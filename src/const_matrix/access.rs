//! ConstMatrix accessors and operations (cross product, norms).

use crate::traits::MatrixElement;
use std::ops::{Mul, Sub};

use super::ConstMatrix;

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
    T: MatrixElement + std::fmt::Debug + crate::traits::ToleranceOps + Sub<Output = T> + Clone,
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
