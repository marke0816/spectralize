//! ConstMatrix arithmetic operations.

use crate::traits::MatrixElement;
use std::ops::{Add, Mul, Sub};

use super::ConstMatrix;

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
// Linear System Solvers for Square Const Matrices
// ============================================================================
