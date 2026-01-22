//! Compile-time dimension matrices using const generics.
//!
//! This module provides `ConstMatrix<T, R, C>` where dimensions are known at compile time,
//! enabling the type system to enforce dimension compatibility for operations.
//!
//! # Examples
//!
//! ```
//! use spectralize::matrix::{ConstMatrix, Matrix};
//!
//! // Create a 3x3 identity matrix - dimensions are in the type
//! let identity: ConstMatrix<f64, 3, 3> = ConstMatrix::identity();
//!
//! // Addition requires matching dimensions (enforced at compile time)
//! let a: ConstMatrix<f64, 2, 2> = ConstMatrix::identity();
//! let b: ConstMatrix<f64, 2, 2> = ConstMatrix::identity();
//! let c = a + b; // ✓ Compiles
//!
//! // Multiplication enforces inner dimension compatibility
//! let d: ConstMatrix<f64, 2, 3> = ConstMatrix::zero();
//! let e: ConstMatrix<f64, 3, 4> = ConstMatrix::zero();
//! let f: ConstMatrix<f64, 2, 4> = d * e; // ✓ Compiles - dimensions match
//!
//! // Convert to/from dynamic Matrix
//! let dyn_mat: Matrix<f64> = identity.into(); // Always succeeds
//! let const_mat: ConstMatrix<f64, 3, 3> = dyn_mat.try_into()?; // Fallible
//! # Ok::<(), spectralize::matrix::MatrixError>(())
//! ```

use super::MatrixElement;
use crate::traits::ToleranceOps;
use std::ops::{Add, Mul};

/// A matrix with compile-time known dimensions.
///
/// Unlike `Matrix<T>` where dimensions are runtime values, `ConstMatrix<T, R, C>`
/// encodes dimensions in the type system. This provides:
/// - Compile-time dimension compatibility checking
/// - No runtime dimension validation overhead
/// - Type-safe operations that cannot fail due to dimension mismatches
///
/// # Storage
///
/// Uses `Vec<T>` in row-major order with the invariant `data.len() == R * C`.
/// While heap-allocated, this avoids stack overflow for large matrices and
/// works with non-Copy types (like `Complex<f64>`).
///
/// # Type Parameters
///
/// - `T`: Element type, must implement `MatrixElement + Debug`
/// - `R`: Number of rows (compile-time constant)
/// - `C`: Number of columns (compile-time constant)
#[derive(Debug, Clone, PartialEq)]
pub struct ConstMatrix<T: MatrixElement + std::fmt::Debug, const R: usize, const C: usize> {
    data: Vec<T>,
}

fn scaled_tolerance<T, const N: usize>(data: &[T]) -> T::Abs
where
    T: super::ToleranceOps,
    T::Abs: MatrixElement + Add<Output = T::Abs> + Mul<Output = T::Abs> + PartialOrd,
{
    match T::epsilon() {
        Some(eps) => {
            let mut max_sum = T::Abs::zero();
            for row in 0..N {
                let mut sum = T::Abs::zero();
                for col in 0..N {
                    sum = sum + data[row * N + col].abs_val();
                }
                if sum > max_sum {
                    max_sum = sum;
                }
            }

            let mut n_factor = T::Abs::one();
            for _ in 1..N {
                n_factor = n_factor + T::Abs::one();
            }

            n_factor * (eps * max_sum)
        }
        None => T::Abs::zero(),
    }
}

// Module declarations
pub mod access;
pub mod arithmetic;
pub mod constructors;
pub mod conversions;
pub mod linalg;
