//! Error types for matrix operations.

/// Errors that can occur during matrix operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixError {
    /// Matrix dimensions are incompatible for the requested operation.
    DimensionMismatch,
    /// Index is out of bounds for the matrix dimensions.
    IndexOutOfBounds,
    /// Permutation vector contains duplicate indices.
    PermutationDuplicateIndex,
    /// Permutation index is out of valid range.
    PermutationIndexOutOfBounds,
    /// Permutation vector length doesn't match expected dimensions.
    PermutationLengthMismatch,
}
