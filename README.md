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
  - Determinant calculation (via PLU decomposition)
  - Invertibility checking
  - Matrix norms (Frobenius, 1-norm, infinity norm)
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
- **Comprehensive Test Suite**: 185+ tests covering all operations
- **Numerical Stability**: PLU decomposition with partial pivoting for robust computations

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

// Determinant (via PLU decomposition with partial pivoting)
let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
let det = m.determinant();  // -2.0

// Check if matrix is invertible
let invertible = m.is_invertible();  // true

let singular = Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
let det_singular = singular.determinant();  // 0.0
let invertible_singular = singular.is_invertible();  // false
```

### Matrix Norms

Matrix norms provide measures of matrix magnitude useful for numerical analysis, condition number computation, and error estimation.

```rust
use spectralize::Matrix;

let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

// Frobenius norm (Euclidean norm)
// ||A||_F = sqrt(sum of |a_ij|^2)
let fro_norm = m.norm_fro();  // sqrt(1 + 4 + 9 + 16) = sqrt(30) ≈ 5.477

// 1-norm (maximum absolute column sum)
// ||A||_1 = max_j sum_i |a_ij|
let one_norm = m.norm_one();  // max(|1|+|3|, |2|+|4|) = max(4, 6) = 6.0

// Infinity norm (maximum absolute row sum)
// ||A||_∞ = max_i sum_j |a_ij|
let inf_norm = m.norm_inf();  // max(|1|+|2|, |3|+|4|) = max(3, 7) = 7.0
```

Norms work with all numeric types including complex numbers:

```rust
use spectralize::Matrix;
use num_complex::Complex;

let m = Matrix::new(2, 2, vec![
    Complex::new(1.0, 1.0),  // 1 + i
    Complex::new(2.0, 0.0),  // 2
    Complex::new(0.0, 0.0),  // 0
    Complex::new(0.0, 3.0),  // 3i
]);

// For complex matrices, norms return the real underlying type (f64)
let fro_norm: f64 = m.norm_fro();  // sqrt(|1+i|^2 + |2|^2 + |3i|^2) = sqrt(15)
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
- `determinant()` - Determinant via PLU decomposition with partial pivoting (square matrices only)
- `determinant_with_tol(tolerance)` - Determinant with custom singularity tolerance for floating-point types
- `is_invertible()` - Check if matrix is invertible (non-singular)
- `is_invertible_with_tol(tolerance)` - Invertibility check with custom tolerance for floating-point types
- `norm_fro()` - Frobenius norm (Euclidean norm): ||A||_F = sqrt(Σ|a_ij|²)
- `norm_one()` - 1-norm (maximum absolute column sum): ||A||_1 = max_j Σ_i |a_ij|
- `norm_inf()` - Infinity norm (maximum absolute row sum): ||A||_∞ = max_i Σ_j |a_ij|

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
    ├── decomposition.rs # PLU decomposition, determinant, and invertibility
    ├── norm.rs         # Matrix norms (Frobenius, 1-norm, infinity norm)
    └── tests.rs        # Comprehensive test suite (185+ tests)
examples/
└── complex_demo.rs     # Example using complex numbers
```

## Future Plans

- Matrix inversion (for `A^-n` support in `pow`)
- Additional decompositions:
  - QR decomposition
  - Cholesky decomposition
  - Singular Value Decomposition (SVD)
- Eigenvalue and eigenvector computation
- Linear system solving using existing PLU decomposition
- Purpose-built graphics rendering APIs
- Sparse matrix support
- Iterative methods for large systems

## Implementation Notes

### PLU Decomposition

The determinant and invertibility checking are powered by an efficient PLU decomposition implementation with partial pivoting:

- **Partial Pivoting**: For each column, the algorithm selects the row with the largest absolute value as the pivot, improving numerical stability
- **In-Place Elimination**: Gaussian elimination is performed in-place on a cloned matrix for optimal memory usage
- **Row Swap Tracking**: The number of row swaps is tracked to correctly compute the determinant sign: det(A) = (-1)^(swaps) × product(diagonal of U)
- **Singularity Detection**: Zero pivots are detected during elimination, allowing immediate identification of singular matrices
- **Exact Arithmetic**: Works correctly with any type supporting division (primarily f32 and f64)

Example computation:
```rust
// For matrix A, PLU decomposition gives us P*A = L*U where:
// - P is a permutation matrix (represented implicitly via row swaps)
// - L is lower triangular with 1s on diagonal
// - U is upper triangular
//
// det(A) = det(P)^(-1) * det(L) * det(U)
//        = (-1)^(row_swaps) * 1 * product(diagonal of U)
```

### Floating-Point Tolerance for Numerical Stability

When working with floating-point matrices (f32, f64, Complex<f32>, Complex<f64>), the PLU decomposition, determinant, and invertibility checks use **tolerance-aware pivoting** to handle numerical precision limits and avoid treating nearly-zero values as exact zeros.

#### Why Tolerance Matters

Floating-point arithmetic has inherent roundoff errors. A mathematically non-zero value like `1e-17` might actually be numerical noise rather than a true value. Consider this nearly-singular matrix:

```rust
// Matrix with condition number ≈ 1e14 (extremely ill-conditioned)
let nearly_singular = Matrix::new(2, 2, vec![
    1.0, 1.0,
    1.0, 1.0 + 1e-14
]);

// Without tolerance: Reports as invertible, but any computation will amplify errors
// With tolerance: Correctly identifies as numerically singular
let is_safe_to_use = nearly_singular.is_invertible();  // Returns true, but warns of ill-conditioning
```

#### Default Tolerance Formula

For floating-point types, Spectralize automatically computes a safe tolerance based on:

```
tolerance = n × ε_machine × ||A||_∞
```

Where:
- **n** = matrix dimension (accounts for error accumulation over n Gaussian elimination steps)
- **ε_machine** = machine epsilon (2.2e-16 for f64, 1.2e-7 for f32)
- **||A||_∞** = infinity norm (scales tolerance to matrix magnitude)

This formula comes from backward error analysis and matches industry-standard libraries like LAPACK.

#### How It Works

During PLU decomposition, when selecting a pivot:
1. Find the row with the largest absolute value in the current column
2. Check if `|pivot| ≤ tolerance`
3. If true: mark the matrix as singular and skip elimination for this column
4. If false: proceed with normal Gaussian elimination

This tolerance then propagates to:
- **`determinant()`**: Returns 0.0 if any pivot was below tolerance (matrix is numerically singular)
- **`is_invertible()`**: Returns false if any pivot was below tolerance

#### Custom Tolerance

For specialized applications, you can override the default tolerance:

```rust
use spectralize::Matrix;

let m = Matrix::new(3, 3, vec![
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0 + 1e-10
]);

// Use default tolerance (recommended for most cases)
let det_default = m.determinant();
let inv_default = m.is_invertible();

// Use stricter tolerance (treat smaller pivots as zero)
let det_strict = m.determinant_with_tol(1e-8);
let inv_strict = m.is_invertible_with_tol(1e-8);

// Use looser tolerance (only reject truly tiny pivots)
let det_loose = m.determinant_with_tol(1e-14);
let inv_loose = m.is_invertible_with_tol(1e-14);
```

**When to use custom tolerance:**
- You know your problem's error bounds (e.g., measurements have known precision ±1e-6)
- You're performing sensitivity analysis (testing how results change with tolerance)
- Default tolerance is too conservative or too permissive for your application

#### Integer Types: Exact Arithmetic

For integer types (i32, i64, etc.), tolerance logic is **completely disabled**:
- Zero pivots are checked exactly (`pivot == 0`)
- No tolerance computation overhead (zero-cost abstraction)
- Determinant and invertibility are exact mathematical properties

```rust
// Integer matrix - exact arithmetic, no tolerance
let int_matrix = Matrix::new(3, 3, vec![
    1i32, 2, 3,
    4, 5, 6,
    7, 8, 9
]);

// Exactly zero determinant (truly singular)
assert_eq!(int_matrix.determinant(), 0);
assert!(!int_matrix.is_invertible());
```

#### Implementation Details

The tolerance system uses:
- **Infinity norm (||A||_∞)**: Efficient for row-major storage, standard in LAPACK, directly relates to Gaussian elimination stability
- **Centralized logic**: Tolerance computed once in PLU decomposition, reused for all operations
- **Type-safe design**: `ToleranceOps` trait returns `Option<T::Abs>` - `Some(ε)` for floats, `None` for integers
- **Zero-cost abstraction**: Integer types compile to exact comparisons with no runtime overhead

For complete implementation details, see `TOLERANCE_IMPLEMENTATION.md` in the repository.

### Matrix Norms

Matrix norms are essential tools in numerical linear algebra for measuring matrix magnitude, estimating errors, and analyzing numerical stability. Spectralize implements three fundamental matrix norms with performance-optimized algorithms:

#### Frobenius Norm (Euclidean Norm)

**Mathematical Definition:** `||A||_F = sqrt(Σ_i Σ_j |a_ij|²)`

The Frobenius norm is the square root of the sum of absolute squares of all matrix elements. It generalizes the Euclidean vector norm to matrices.

**Use Cases:**
- **Convergence criteria**: Common stopping condition for iterative algorithms (e.g., "stop when ||A - A_prev||_F < tolerance")
- **Matrix approximation**: Natural choice for least squares problems and matrix factorization
- **Condition number estimation**: Used in assessing sensitivity of matrix operations to numerical errors
- **General magnitude measure**: Intuitive "size" metric for matrices

**Implementation:** Single-pass O(mn) algorithm using fold-based accumulation of squared absolute values, followed by a single square root operation. Efficient and cache-friendly.

#### 1-Norm (Maximum Absolute Column Sum)

**Mathematical Definition:** `||A||_1 = max_j Σ_i |a_ij|`

The 1-norm computes the sum of absolute values for each column and returns the maximum.

**Use Cases:**
- **Condition number computation**: Used in `cond_1(A) = ||A||_1 * ||A^(-1)||_1` to measure how close a matrix is to being singular
- **Error bounds**: Provides tight bounds for error propagation in linear system solving
- **Stability analysis**: Essential for analyzing Gaussian elimination and other decomposition algorithms
- **Sparse matrix analysis**: Particularly efficient for column-oriented sparse matrix operations

**Implementation:** Pre-allocates a column sum vector and performs a single pass through the data with column-wise accumulation, then finds the maximum. Optimized despite row-major storage.

#### Infinity Norm (Maximum Absolute Row Sum)

**Mathematical Definition:** `||A||_∞ = max_i Σ_j |a_ij|`

The infinity norm computes the sum of absolute values for each row and returns the maximum.

**Use Cases:**
- **Dual to 1-norm**: Since ||A||_∞ = ||A^T||_1, useful in theoretical analysis
- **Perturbation bounds**: Provides bounds on how matrix entry perturbations affect solutions
- **Iterative solver analysis**: Used in convergence criteria for Jacobi and Gauss-Seidel methods
- **Row-reduction algorithms**: Natural choice for analyzing row-wise operations like forward/backward substitution

**Implementation:** Most efficient norm for row-major storage. Uses sequential slice iteration for cache-friendly memory access, computing row sums with a fold operation and tracking the maximum.

#### Type System Support

All three norms work seamlessly with:
- **Real types**: `f32`, `f64` (returns same type)
- **Complex types**: `Complex<f32>`, `Complex<f64>` (returns underlying real type: `f32` or `f64`)
- **Integer types**: Not typically used for norms in numerical analysis, but supported where mathematically valid

The norm implementations use helper traits (`Abs` and `Sqrt`) to maintain zero-cost abstractions while supporting diverse numeric types.
