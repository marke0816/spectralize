//! ConstMatrix linear algebra operations.

use crate::error::MatrixError;
use crate::matrix::Matrix;
use crate::traits::{MatrixElement, Sqrt};
use std::ops::{Add, Div, Mul, Neg, Sub};

use super::{scaled_tolerance, ConstMatrix};

impl<T, const R: usize, const C: usize> ConstMatrix<T, R, C>
where
    T: MatrixElement + std::fmt::Debug + crate::traits::Abs,
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
        T::Output: crate::traits::Sqrt,
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
        T: Mul<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Neg<Output = T>
            + super::ToleranceOps,
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
        T: Mul<Output = T>
            + Add<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Neg<Output = T>
            + super::ToleranceOps,
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

    /// Compute the inverse of a 3x3 matrix using a custom tolerance threshold.
    ///
    /// This variant allows you to specify your own tolerance for singularity detection.
    /// A matrix is considered singular if |det(A)| ≤ tolerance.
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
    /// let inv = m.inverse_with_tol(1e-10).unwrap();
    /// // Verify: M * M^-1 ≈ I
    /// let identity = &m * &inv;
    /// assert!(identity.approx_eq(&ConstMatrix::identity(), 1e-9));
    /// ```
    pub fn inverse_with_tol(&self, tolerance: T::Abs) -> Option<ConstMatrix<T, 3, 3>>
    where
        T: Mul<Output = T>
            + Add<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Neg<Output = T>
            + super::ToleranceOps,
        T::Abs: PartialOrd,
    {
        let det = self.determinant();

        // Check if determinant is (near) zero using custom tolerance
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
        T: Mul<Output = T>
            + Add<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Neg<Output = T>
            + super::ToleranceOps,
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
                let sign = if (i + j) % 2 == 0 {
                    T::one()
                } else {
                    -T::one()
                };
                data.push((sign * cofactor) / det.clone());
            }
        }

        Some(ConstMatrix { data })
    }

    /// Compute the inverse of a 4x4 matrix using a custom tolerance threshold.
    ///
    /// This variant allows you to specify your own tolerance for singularity detection.
    /// A matrix is considered singular if |det(A)| ≤ tolerance.
    ///
    /// # Examples
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let m: ConstMatrix<f64, 4, 4> = ConstMatrix::identity();
    /// let inv = m.inverse_with_tol(1e-10).unwrap();
    /// assert!(inv.approx_eq(&ConstMatrix::identity(), 1e-10));
    /// ```
    pub fn inverse_with_tol(&self, tolerance: T::Abs) -> Option<ConstMatrix<T, 4, 4>>
    where
        T: Mul<Output = T>
            + Add<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Neg<Output = T>
            + super::ToleranceOps,
        T::Abs: PartialOrd,
    {
        let det = self.determinant();

        // Check if determinant is (near) zero using custom tolerance
        if det.abs_val() <= tolerance {
            return None;
        }

        // Compute adjugate matrix (transpose of cofactor matrix)
        let mut data = Vec::with_capacity(16);

        for j in 0..4 {
            for i in 0..4 {
                let cofactor = self.cofactor_3x3(i, j);
                let sign = if (i + j) % 2 == 0 {
                    T::one()
                } else {
                    -T::one()
                };
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
impl<T, const N: usize> ConstMatrix<T, N, N>
where
    T: MatrixElement + std::fmt::Debug,
{
    /// Solve the linear system Ax = b using default tolerance.
    ///
    /// Returns the solution matrix X where A is this matrix (N×N) and b is the
    /// right-hand side (N×K). For small matrices (N ≤ 4), uses fast closed-form
    /// inverses. For larger matrices, uses PLU decomposition from dynamic Matrix.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is singular (non-invertible).
    ///
    /// # Example
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let a: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0]);
    /// let b: ConstMatrix<f64, 2, 1> = ConstMatrix::new(vec![5.0, 11.0]);
    /// let x = a.solve(&b);
    ///
    /// // Verify: x should be [1.0, 2.0]
    /// assert!((x.get(0, 0) - 1.0f64).abs() < 1e-10);
    /// assert!((x.get(1, 0) - 2.0f64).abs() < 1e-10);
    /// ```
    pub fn solve<const K: usize>(&self, b: &ConstMatrix<T, N, K>) -> ConstMatrix<T, N, K>
    where
        T: super::ToleranceOps
            + crate::traits::PivotOrd
            + Div<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + Add<Output = T>
            + Neg<Output = T>
            + Clone
            + crate::traits::Abs<Output = T::Abs>,
        T::Abs: MatrixElement
            + Add<Output = T::Abs>
            + Mul<Output = T::Abs>
            + PartialOrd
            + crate::traits::NanCheck,
    {
        self.try_solve(b).expect("Matrix must be invertible")
    }

    /// Solve the linear system Ax = b using default tolerance.
    ///
    /// This is the checked version that returns a `Result` instead of panicking.
    ///
    /// # Errors
    ///
    /// Returns `MatrixError::DimensionMismatch` if the matrix is singular.
    ///
    /// # Example
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let a: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0]);
    /// let b: ConstMatrix<f64, 2, 1> = ConstMatrix::new(vec![5.0, 11.0]);
    /// let x = a.try_solve(&b).unwrap();
    ///
    /// // Verify: x should be [1.0, 2.0]
    /// assert!((x.get(0, 0) - 1.0f64).abs() < 1e-10);
    /// assert!((x.get(1, 0) - 2.0f64).abs() < 1e-10);
    /// ```
    pub fn try_solve<const K: usize>(
        &self,
        b: &ConstMatrix<T, N, K>,
    ) -> Result<ConstMatrix<T, N, K>, MatrixError>
    where
        T: super::ToleranceOps
            + crate::traits::PivotOrd
            + Div<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + Add<Output = T>
            + Neg<Output = T>
            + Clone
            + crate::traits::Abs<Output = T::Abs>,
        T::Abs: MatrixElement
            + Add<Output = T::Abs>
            + Mul<Output = T::Abs>
            + PartialOrd
            + crate::traits::NanCheck,
    {
        // Delegate to the with_tol version using default tolerance computation
        if N <= 4 {
            // For small matrices, use scaled_tolerance
            let tol = scaled_tolerance::<T, N>(&self.data);
            self.try_solve_with_tol_impl(b, tol)
        } else {
            // For large matrices, use dynamic Matrix solver which handles tolerance internally
            self.try_solve_dynamic(b)
        }
    }

    /// Solve the linear system Ax = b using a custom tolerance.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is singular under the specified tolerance.
    ///
    /// # Example
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let a: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0]);
    /// let b: ConstMatrix<f64, 2, 1> = ConstMatrix::new(vec![5.0, 11.0]);
    /// let x = a.solve_with_tol(&b, 1e-10);
    /// ```
    pub fn solve_with_tol<const K: usize>(
        &self,
        b: &ConstMatrix<T, N, K>,
        tolerance: T::Abs,
    ) -> ConstMatrix<T, N, K>
    where
        T: super::ToleranceOps
            + crate::traits::PivotOrd
            + Div<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + Add<Output = T>
            + Neg<Output = T>
            + Clone
            + crate::traits::Abs<Output = T::Abs>,
        T::Abs: MatrixElement
            + Add<Output = T::Abs>
            + Mul<Output = T::Abs>
            + PartialOrd
            + crate::traits::NanCheck,
    {
        self.try_solve_with_tol(b, tolerance)
            .expect("Matrix must be invertible")
    }

    /// Solve the linear system Ax = b using a custom tolerance.
    ///
    /// This is the checked version that returns a `Result` instead of panicking.
    ///
    /// # Errors
    ///
    /// Returns `MatrixError::DimensionMismatch` if the matrix is singular
    /// under the specified tolerance.
    ///
    /// # Example
    ///
    /// ```
    /// use spectralize::matrix::ConstMatrix;
    ///
    /// let a: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0]);
    /// let b: ConstMatrix<f64, 2, 1> = ConstMatrix::new(vec![5.0, 11.0]);
    /// let x = a.try_solve_with_tol(&b, 1e-10).unwrap();
    /// ```
    pub fn try_solve_with_tol<const K: usize>(
        &self,
        b: &ConstMatrix<T, N, K>,
        tolerance: T::Abs,
    ) -> Result<ConstMatrix<T, N, K>, MatrixError>
    where
        T: super::ToleranceOps
            + crate::traits::PivotOrd
            + Div<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + Add<Output = T>
            + Neg<Output = T>
            + Clone
            + crate::traits::Abs<Output = T::Abs>,
        T::Abs: MatrixElement
            + Add<Output = T::Abs>
            + Mul<Output = T::Abs>
            + PartialOrd
            + crate::traits::NanCheck,
    {
        if N <= 4 {
            self.try_solve_with_tol_impl(b, tolerance)
        } else {
            self.try_solve_dynamic_with_tol(b, tolerance)
        }
    }

    /// Internal implementation for N ≤ 4 using closed-form inverses
    fn try_solve_with_tol_impl<const K: usize>(
        &self,
        b: &ConstMatrix<T, N, K>,
        tolerance: T::Abs,
    ) -> Result<ConstMatrix<T, N, K>, MatrixError>
    where
        T: Div<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + Add<Output = T>
            + Neg<Output = T>
            + super::ToleranceOps,
        T::Abs: MatrixElement + Add<Output = T::Abs> + Mul<Output = T::Abs> + PartialOrd,
    {
        match N {
            1 => self.try_solve_1x1(b, tolerance),
            2 => self.try_solve_2x2(b, tolerance),
            3 => self.try_solve_3x3(b, tolerance),
            4 => self.try_solve_4x4(b, tolerance),
            _ => unreachable!("N <= 4 constraint violated"),
        }
    }

    /// Solve 1x1 system: x = b / a
    fn try_solve_1x1<const K: usize>(
        &self,
        b: &ConstMatrix<T, N, K>,
        tolerance: T::Abs,
    ) -> Result<ConstMatrix<T, N, K>, MatrixError>
    where
        T: Div<Output = T> + super::ToleranceOps,
        T::Abs: PartialOrd,
    {
        let a = &self.data[0];

        // Check singularity
        if a.abs_val() <= tolerance {
            return Err(MatrixError::DimensionMismatch);
        }

        // Solve: x = b / a for each column
        let mut data = Vec::with_capacity(K);
        for k in 0..K {
            data.push(b.data[k].clone() / a.clone());
        }

        Ok(ConstMatrix { data })
    }

    /// Solve 2x2 system using inverse
    fn try_solve_2x2<const K: usize>(
        &self,
        b: &ConstMatrix<T, N, K>,
        tolerance: T::Abs,
    ) -> Result<ConstMatrix<T, N, K>, MatrixError>
    where
        T: Mul<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Add<Output = T>
            + Neg<Output = T>
            + super::ToleranceOps,
        T::Abs: PartialOrd,
    {
        // Compute determinant
        let det = (&self.data[0]).clone() * (&self.data[3]).clone()
            - (&self.data[1]).clone() * (&self.data[2]).clone();

        // Check singularity
        if det.abs_val() <= tolerance {
            return Err(MatrixError::DimensionMismatch);
        }

        // Compute inverse elements: [[d, -b], [-c, a]] / det
        let a = &self.data[0];
        let b_elem = &self.data[1];
        let c = &self.data[2];
        let d = &self.data[3];

        let inv00 = d.clone() / det.clone();
        let inv01 = -(b_elem.clone()) / det.clone();
        let inv10 = -(c.clone()) / det.clone();
        let inv11 = a.clone() / det.clone();

        // Multiply inverse by b: X = A^-1 * B
        let mut data = Vec::with_capacity(2 * K);
        for col in 0..K {
            let b0 = &b.data[col];
            let b1 = &b.data[K + col];

            data.push(inv00.clone() * b0.clone() + inv01.clone() * b1.clone());
            data.push(inv10.clone() * b0.clone() + inv11.clone() * b1.clone());
        }

        Ok(ConstMatrix { data })
    }

    /// Solve 3x3 system using inverse
    fn try_solve_3x3<const K: usize>(
        &self,
        b: &ConstMatrix<T, N, K>,
        tolerance: T::Abs,
    ) -> Result<ConstMatrix<T, N, K>, MatrixError>
    where
        T: Mul<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Add<Output = T>
            + Neg<Output = T>
            + super::ToleranceOps,
        T::Abs: PartialOrd,
    {
        // Cast self to ConstMatrix<T, 3, 3> and use inverse_with_tol
        // This is safe because we know N == 3
        let self_3x3: &ConstMatrix<T, 3, 3> =
            unsafe { &*(self as *const _ as *const ConstMatrix<T, 3, 3>) };
        let inv = self_3x3
            .inverse_with_tol(tolerance)
            .ok_or(MatrixError::DimensionMismatch)?;

        // Cast back to N=3 for multiplication
        let inv_nxn: &ConstMatrix<T, N, N> =
            unsafe { &*(&inv as *const _ as *const ConstMatrix<T, N, N>) };
        Ok(inv_nxn * b)
    }

    /// Solve 4x4 system using inverse
    fn try_solve_4x4<const K: usize>(
        &self,
        b: &ConstMatrix<T, N, K>,
        tolerance: T::Abs,
    ) -> Result<ConstMatrix<T, N, K>, MatrixError>
    where
        T: Mul<Output = T>
            + Sub<Output = T>
            + Div<Output = T>
            + Add<Output = T>
            + Neg<Output = T>
            + super::ToleranceOps,
        T::Abs: PartialOrd,
    {
        // Cast self to ConstMatrix<T, 4, 4> and use inverse_with_tol
        let self_4x4: &ConstMatrix<T, 4, 4> =
            unsafe { &*(self as *const _ as *const ConstMatrix<T, 4, 4>) };
        let inv = self_4x4
            .inverse_with_tol(tolerance)
            .ok_or(MatrixError::DimensionMismatch)?;

        // Cast back to N=4 for multiplication
        let inv_nxn: &ConstMatrix<T, N, N> =
            unsafe { &*(&inv as *const _ as *const ConstMatrix<T, N, N>) };
        Ok(inv_nxn * b)
    }

    /// Solve for N > 4 using dynamic Matrix solver (default tolerance)
    fn try_solve_dynamic<const K: usize>(
        &self,
        b: &ConstMatrix<T, N, K>,
    ) -> Result<ConstMatrix<T, N, K>, MatrixError>
    where
        T: super::ToleranceOps
            + crate::traits::PivotOrd
            + Div<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + Add<Output = T>
            + Neg<Output = T>
            + Clone
            + crate::traits::Abs<Output = T::Abs>,
        T::Abs: MatrixElement
            + Add<Output = T::Abs>
            + Mul<Output = T::Abs>
            + PartialOrd
            + crate::traits::NanCheck,
    {
        // Convert to dynamic Matrix
        let mut a_data = Vec::with_capacity(N * N);
        for i in 0..N {
            for j in 0..N {
                a_data.push(self.data[i * N + j].clone());
            }
        }
        let a_dyn = Matrix::new(N, N, a_data);

        let mut b_data = Vec::with_capacity(N * K);
        for i in 0..N {
            for j in 0..K {
                b_data.push(b.data[i * K + j].clone());
            }
        }
        let b_dyn = Matrix::new(N, K, b_data);

        // Solve using dynamic Matrix API
        let x_dyn = a_dyn.try_solve_matrix(&b_dyn)?;

        // Convert back to ConstMatrix
        x_dyn.try_into()
    }

    /// Solve for N > 4 using dynamic Matrix solver (custom tolerance)
    fn try_solve_dynamic_with_tol<const K: usize>(
        &self,
        b: &ConstMatrix<T, N, K>,
        tolerance: T::Abs,
    ) -> Result<ConstMatrix<T, N, K>, MatrixError>
    where
        T: super::ToleranceOps
            + crate::traits::PivotOrd
            + Div<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + Add<Output = T>
            + Neg<Output = T>
            + Clone
            + crate::traits::Abs<Output = T::Abs>,
        T::Abs: MatrixElement
            + Add<Output = T::Abs>
            + Mul<Output = T::Abs>
            + PartialOrd
            + crate::traits::NanCheck,
    {
        // Convert to dynamic Matrix
        let mut a_data = Vec::with_capacity(N * N);
        for i in 0..N {
            for j in 0..N {
                a_data.push(self.data[i * N + j].clone());
            }
        }
        let a_dyn = Matrix::new(N, N, a_data);

        let mut b_data = Vec::with_capacity(N * K);
        for i in 0..N {
            for j in 0..K {
                b_data.push(b.data[i * K + j].clone());
            }
        }
        let b_dyn = Matrix::new(N, K, b_data);

        // Solve using dynamic Matrix API with tolerance
        let x_dyn = a_dyn.try_solve_matrix_with_tol(&b_dyn, tolerance)?;

        // Convert back to ConstMatrix
        x_dyn.try_into()
    }
}

// ============================================================================
// Conversions
// ============================================================================

// Conversion from ConstMatrix to dynamic Matrix (always succeeds)
