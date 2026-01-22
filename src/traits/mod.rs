//! Traits for matrix element types and numerical operations.

pub mod element;
pub mod norm;
pub mod pivot;
pub mod tolerance;

pub use element::MatrixElement;
pub use norm::{Abs, Sqrt};
pub use pivot::PivotOrd;
pub use tolerance::{NanCheck, ToleranceOps};
