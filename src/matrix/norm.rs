use super::{Matrix, MatrixElement};
use std::ops::{Add, Mul};

/// Trait for types that support absolute value operations.
/// Required for computing matrix norms that depend on element magnitudes.
pub trait Abs {
    /// The output type of the absolute value operation.
    /// For real numbers, this is Self. For complex numbers, this is the underlying real type.
    type Output;

    /// Compute the absolute value (magnitude) of this element.
    fn abs(self) -> Self::Output;
}

/// Trait for types that support square root operations.
/// Required for computing the Frobenius norm.
pub trait Sqrt {
    /// Compute the square root of this value.
    fn sqrt(self) -> Self;
}

// Implementations for f32
impl Abs for f32 {
    type Output = f32;

    fn abs(self) -> Self::Output {
        f32::abs(self)
    }
}

impl Sqrt for f32 {
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}

// Implementations for f64
impl Abs for f64 {
    type Output = f64;

    fn abs(self) -> Self::Output {
        f64::abs(self)
    }
}

impl Sqrt for f64 {
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

// Implementations for Complex<f32>
impl Abs for num_complex::Complex<f32> {
    type Output = f32;

    fn abs(self) -> Self::Output {
        self.norm()
    }
}

// Implementations for Complex<f64>
impl Abs for num_complex::Complex<f64> {
    type Output = f64;

    fn abs(self) -> Self::Output {
        self.norm()
    }
}

impl<T> Matrix<T>
where
    T: MatrixElement + std::fmt::Debug,
{
    /// Compute the Frobenius norm (Euclidean norm) of the matrix.
    ///
    /// **Mathematical Definition:**
    /// ||A||_F = sqrt(sum_{i,j} |a_{ij}|^2)
    ///
    /// The Frobenius norm is the square root of the sum of the absolute squares
    /// of all matrix elements. It is equivalent to treating the matrix as a vector
    /// and computing its Euclidean (L2) norm.
    ///
    /// **Why This Norm is Useful:**
    /// - **Natural measure of matrix magnitude**: Provides an intuitive sense of
    ///   the "size" of a matrix that generalizes the Euclidean norm from vectors.
    /// - **Condition number estimation**: Used in estimating how sensitive matrix
    ///   operations (like inversion) are to numerical errors.
    /// - **Convergence criteria**: Common stopping criterion for iterative algorithms
    ///   (e.g., "stop when ||A - A_prev||_F < tolerance").
    /// - **Least squares**: The Frobenius norm appears naturally in matrix approximation
    ///   problems and least squares formulations.
    /// - **Computationally efficient**: Can be computed in a single pass through
    ///   all matrix elements with O(mn) complexity.
    ///
    /// **Implementation:**
    /// This implementation runs in a single pass over all elements, accumulating
    /// the sum of squares, then computing the square root at the end. This is
    /// more efficient than repeated intermediate square root operations.
    pub fn norm_fro(&self) -> T::Output
    where
        T: Clone + Abs,
        T::Output: MatrixElement + Mul<Output = T::Output> + Add<Output = T::Output> + Sqrt,
    {
        // Single-pass accumulation of sum of squares
        let sum_of_squares = self.data.iter().fold(T::Output::zero(), |acc, element| {
            let abs_val = element.clone().abs();
            acc + (abs_val.clone() * abs_val)
        });

        sum_of_squares.sqrt()
    }

    /// Compute the 1-norm (maximum absolute column sum) of the matrix.
    ///
    /// **Mathematical Definition:**
    /// ||A||_1 = max_j sum_i |a_{ij}|
    ///
    /// The 1-norm is the maximum of the sums of absolute values of each column.
    /// For each column j, we sum the absolute values of all elements in that column,
    /// then take the maximum across all columns.
    ///
    /// **Why This Norm is Useful:**
    /// - **Condition number bounds**: The 1-norm provides computable bounds for
    ///   the condition number, which measures how errors are amplified in matrix
    ///   operations like solving linear systems.
    /// - **Stability analysis**: Used in numerical analysis to bound error propagation
    ///   in algorithms like Gaussian elimination.
    /// - **Matrix inversion**: The 1-norm of A and A^(-1) are used together to
    ///   compute cond_1(A) = ||A||_1 * ||A^(-1)||_1, which indicates how close
    ///   a matrix is to being singular.
    /// - **Sparse matrices**: Particularly useful for sparse matrix analysis where
    ///   column operations are common.
    /// - **Easy to compute**: Requires only one pass through the matrix elements
    ///   with column-wise accumulation.
    ///
    /// **Implementation:**
    /// Iterates through each column, accumulating the sum of absolute values,
    /// and maintains the running maximum. For row-major storage, this requires
    /// strided access patterns, but we optimize by iterating through the data
    /// once and accumulating into column sums.
    pub fn norm_one(&self) -> T::Output
    where
        T: Clone + Abs,
        T::Output: MatrixElement + Add<Output = T::Output> + PartialOrd,
    {
        // Pre-allocate column sums for efficiency
        let mut col_sums = vec![T::Output::zero(); self.cols];

        // Single pass through data: accumulate absolute values by column
        for row in 0..self.rows {
            for col in 0..self.cols {
                let abs_val = self.data[row * self.cols + col].clone().abs();
                col_sums[col] = col_sums[col].clone() + abs_val;
            }
        }

        // Find maximum column sum
        col_sums
            .into_iter()
            .fold(T::Output::zero(), |max_so_far, col_sum| {
                if col_sum > max_so_far {
                    col_sum
                } else {
                    max_so_far
                }
            })
    }

    /// Compute the infinity norm (maximum absolute row sum) of the matrix.
    ///
    /// **Mathematical Definition:**
    /// ||A||_∞ = max_i sum_j |a_{ij}|
    ///
    /// The infinity norm is the maximum of the sums of absolute values of each row.
    /// For each row i, we sum the absolute values of all elements in that row,
    /// then take the maximum across all rows.
    ///
    /// **Why This Norm is Useful:**
    /// - **Dual to 1-norm**: The infinity norm of A equals the 1-norm of A^T,
    ///   which is useful in theoretical analysis and algorithm design.
    /// - **Condition number computation**: Like the 1-norm, used to compute
    ///   cond_∞(A) = ||A||_∞ * ||A^(-1)||_∞ for numerical stability analysis.
    /// - **Perturbation bounds**: Provides tight bounds on how perturbations
    ///   in the matrix entries affect solutions to linear systems.
    /// - **Iterative methods**: Used in convergence analysis of iterative solvers
    ///   like Jacobi and Gauss-Seidel methods.
    /// - **Optimal for row operations**: Natural choice when analyzing algorithms
    ///   that operate row-wise (e.g., row-reduction, forward/backward substitution).
    /// - **Memory-efficient computation**: For row-major storage, this norm can
    ///   be computed very efficiently with sequential memory access.
    ///
    /// **Implementation:**
    /// Takes advantage of row-major storage for cache-friendly sequential access.
    /// For each row, we sum the absolute values of elements, then track the maximum.
    /// This is the most efficient norm to compute in row-major layout.
    pub fn norm_inf(&self) -> T::Output
    where
        T: Clone + Abs,
        T::Output: MatrixElement + Add<Output = T::Output> + PartialOrd,
    {
        let mut max_row_sum = T::Output::zero();

        // Iterate row by row (cache-friendly for row-major storage)
        for row in 0..self.rows {
            // Compute sum of absolute values for this row
            let row_start = row * self.cols;
            let row_end = row_start + self.cols;
            let row_sum = self.data[row_start..row_end]
                .iter()
                .fold(T::Output::zero(), |acc, element| {
                    acc + element.clone().abs()
                });

            // Update maximum if this row sum is larger
            if row_sum > max_row_sum {
                max_row_sum = row_sum;
            }
        }

        max_row_sum
    }
}
