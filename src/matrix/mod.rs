//! Dynamic matrix type with runtime dimensions.

pub mod access;
pub mod append;
pub mod arithmetic;
pub mod constructors;
pub mod core;
pub mod norms;

pub use core::Matrix;

#[cfg(test)]
mod tests;
