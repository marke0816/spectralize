//! Spectralize: A high-performance matrix library for Rust.

pub mod const_matrix;
pub mod error;
pub mod linalg;
pub mod matrix;
pub mod traits;

// Re-export commonly used types
pub use const_matrix::ConstMatrix;
pub use error::MatrixError;
pub use linalg::PLUDecomposition;
pub use matrix::Matrix;
pub use traits::{MatrixElement, ToleranceOps};
