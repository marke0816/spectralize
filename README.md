# Spectralize

A high-performance, generic matrix library for Rust with support for real, integer, and complex number types.

## Features

- **Fully Generic**: Works with any numeric type including `f32`, `f64`, all integer types (`i8` through `i128`, `u8` through `u128`), and complex numbers
- **Rich Arithmetic Operations**:
  - Matrix addition and subtraction
  - Matrix multiplication
  - Scalar multiplication (both `matrix * scalar` and `scalar * matrix`)
  - Matrix exponentiation (`A^n`)
- **Advanced Operations**:
  - Dot product (inner product)
  - Outer product
  - Matrix transpose
  - Trace (sum of diagonal elements)
- **Matrix Construction**:
  - Zero matrices
  - Identity matrices
  - Permutation matrices
  - Custom matrices from vectors
- **Matrix Concatenation**:
  - Horizontal concatenation (`with_cols`)
  - Vertical concatenation (`with_rows`)
  - Row and column appending
- **Memory Efficient**: Optimized implementations with in-place mutations where possible
- **Type Safe**: Compile-time dimension checking through Rust's type system
- **Comprehensive Test Suite**: 109+ tests covering all operations

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
spectralize = "0.1.0"
num-complex = "0.4"  # Required for complex number support
```

## Quick Start

### Basic Matrix Operations

```rust
use spectralize::Matrix;

// Create a 2x3 matrix
let a = Matrix::new(2, 3, vec![
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0
]);

// Create an identity matrix
let identity = Matrix::<f64>::identity(3, 3);

// Create a zero matrix
let zero = Matrix::<f64>::zero(2, 2);

// Access elements
let element = a.get(0, 1);  // Gets element at row 0, col 1
```

### Arithmetic Operations

```rust
use spectralize::Matrix;

let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

// Matrix addition
let sum = &a + &b;

// Matrix subtraction
let diff = &a - &b;

// Matrix multiplication
let product = &a * &b;

// Scalar multiplication (both directions work!)
let scaled1 = &a * 2.0;
let scaled2 = 2.0 * &a;  // Commutative!

// Matrix exponentiation
let a_squared = a.pow(2);  // A^2
let a_cubed = a.pow(3);    // A^3
```

### Working with Different Types

```rust
use spectralize::Matrix;

// Integer matrices
let int_matrix = Matrix::new(2, 2, vec![1i32, 2, 3, 4]);

// Float matrices
let f32_matrix = Matrix::new(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]);
let f64_matrix = Matrix::new(2, 2, vec![1.0f64, 2.0, 3.0, 4.0]);

// All operations work the same way!
let int_sum = &int_matrix * 3i32;
let float_sum = &f32_matrix * 2.5f32;
```

### Complex Numbers

```rust
use spectralize::Matrix;
use num_complex::Complex;

// Create a complex matrix
let m = Matrix::new(2, 2, vec![
    Complex::new(1.0, 2.0),  // 1 + 2i
    Complex::new(3.0, 4.0),  // 3 + 4i
    Complex::new(5.0, 6.0),  // 5 + 6i
    Complex::new(7.0, 8.0),  // 7 + 8i
]);

// Complex identity matrix
let identity = Matrix::<Complex<f64>>::identity(3, 3);

// All arithmetic operations work with complex numbers!
let scaled = m * Complex::new(2.0, 0.0);
```

### Advanced Operations

```rust
use spectralize::Matrix;

let a = Matrix::new(1, 3, vec![1.0, 2.0, 3.0]);
let b = Matrix::new(1, 3, vec![4.0, 5.0, 6.0]);

// Dot product (inner product)
let dot = a.dot(&b);  // 1*4 + 2*5 + 3*6 = 32

// Outer product
let c = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]);
let d = Matrix::new(2, 1, vec![4.0, 5.0]);
let outer = c.outer(&d);  // Produces a 3x2 matrix

// Transpose
let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let m_t = m.transpose();  // Produces a 3x2 matrix

// Trace (sum of diagonal elements)
let square = Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
let tr = square.trace();  // 1 + 5 + 9 = 15
```

### Matrix Concatenation

```rust
use spectralize::Matrix;

let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

// Horizontal concatenation (add columns)
let horizontal = a.with_cols(&b);  // Results in 2x4 matrix

// Vertical concatenation (add rows)
let vertical = a.with_rows(&b);    // Results in 4x2 matrix

// Append a single row
let with_row = a.with_row_vec(&[9.0, 10.0]);  // Results in 3x2 matrix

// Append a single column
let with_col = a.with_col_vec(&[9.0, 10.0]);  // Results in 2x3 matrix
```

### Permutation Matrices

```rust
use spectralize::Matrix;

// Create a permutation matrix
// The vector [2, 4, 3, 1] means:
// - Row 0 has a 1 in column 1 (2-1)
// - Row 1 has a 1 in column 3 (4-1)
// - Row 2 has a 1 in column 2 (3-1)
// - Row 3 has a 1 in column 0 (1-1)
let perm = Matrix::<f64>::perm(4, 4, vec![2, 4, 3, 1]);
```

## API Documentation

### Core Methods

#### Construction
- `Matrix::new(rows, cols, data)` - Create a matrix from a vector
- `Matrix::zero(rows, cols)` - Create a zero matrix
- `Matrix::identity(rows, cols)` - Create an identity matrix
- `Matrix::perm(rows, cols, perm)` - Create a permutation matrix

#### Access
- `get(row, col)` - Get element at position (safe, panics on out of bounds)
- `set(row, col, value)` - Set element at position (safe, panics on out of bounds)
- `row(index)` - Get a row as a slice
- `col(index)` - Get a column as a vector
- `rows()` - Get number of rows
- `cols()` - Get number of columns

#### Arithmetic (via operator overloading)
- `+` - Matrix addition
- `-` - Matrix subtraction
- `*` - Matrix multiplication or scalar multiplication

#### Advanced Operations
- `pow(n)` - Matrix exponentiation (A^n)
- `dot(other)` - Dot product
- `outer(other)` - Outer product
- `transpose()` - Matrix transpose
- `trace()` - Trace (sum of diagonal elements, square matrices only)

#### Concatenation
- `with_cols(other)` - Horizontal concatenation
- `with_rows(other)` - Vertical concatenation
- `with_row_vec(row)` - Append a row
- `with_col_vec(col)` - Append a column

## Examples

Run the included example to see complex number operations:

```bash
cargo run --example complex_demo
```

## Design Philosophy

### Generic by Design
Spectralize uses Rust's powerful trait system to provide a single, unified API that works with any numeric type. The `MatrixElement` trait defines the minimal requirements for a type to be used in matrices.

### Performance
- In-place mutations are used where possible to avoid unnecessary allocations
- Reference and owned variants of operations are provided for flexibility
- Efficient memory layout with row-major storage

### Safety
- All operations include dimension checking
- Type-safe operations prevent common matrix algebra errors
- Comprehensive test coverage ensures correctness

## Running Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test module
cargo test scalar_multiplication
```

## Project Structure

```
src/
├── lib.rs              # Public API exports
└── matrix/
    ├── mod.rs          # Core Matrix struct and basic operations
    ├── element.rs      # MatrixElement trait and implementations
    ├── arithmetic.rs   # Arithmetic operations (Add, Sub, Mul, etc.)
    ├── append.rs       # Matrix concatenation operations
    └── tests.rs        # Comprehensive test suite
examples/
└── complex_demo.rs     # Example using complex numbers
```

## Future Plans

- Matrix inversion (for `A^-n` support in `pow`)
- Determinant calculation
- LU, QR, and Cholesky decompositions
- Eigenvalue and eigenvector computation
- Purpose-built graphics rendering APIs
- Sparse matrix support
