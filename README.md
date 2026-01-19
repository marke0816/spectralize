# Spectralize

A high-performance, generic matrix library for Rust with support for real, integer, and complex number types.

## Features

- **Compile-Time Dimension Matrices**: `ConstMatrix<T, R, C>` with const generics for zero-overhead dimension checking at compile time
- **Fully Generic**: Works with any numeric type including `f32`, `f64`, all integer types (`i8` through `i128`, `u8` through `u128`), and complex numbers
- **Rich Arithmetic Operations**:
  - Matrix addition and subtraction
  - Matrix multiplication
  - Scalar multiplication (both `matrix * scalar` and `scalar * matrix`)
  - Matrix exponentiation (`A^n` for positive n, `A^(-n)` for negative n with invertible matrices)
- **Advanced Operations**:
  - Dot product (inner product)
  - Outer product
  - Cross product (3D vector cross product)
  - Matrix transpose
  - Trace (sum of diagonal elements)
  - Determinant calculation with fast paths for small matrices (closed-form for n ≤ 4, PLU decomposition for n > 4)
  - Invertibility checking with fast paths for small matrices
  - Matrix inversion with fast paths for small matrices (closed-form for n ≤ 4, PLU decomposition for n > 4)
  - Matrix norms (Frobenius, 1-norm, infinity norm)
  - Specialized operations: closed-form determinants and inverses for both `Matrix` (n ≤ 4) and `ConstMatrix` (2×2, 3×3, 4×4)
- **Matrix Construction**:
  - Zero matrices
  - Identity matrices
  - Permutation matrices
  - Custom matrices from vectors
- **Matrix Concatenation**:
  - Horizontal concatenation (`with_cols`)
  - Vertical concatenation (`with_rows`)
  - Row and column appending
- **Approximate Equality**:
  - `approx_eq` for float/complex comparisons with tolerance
- **Column Iteration**:
  - `col_iter` for zero-allocation column access
- **Checked APIs**:
  - Non-panicking `try_*` variants for indexing, concatenation, and core ops
  - `MatrixError` for dimension/index errors
- **Memory Efficient**: Optimized implementations with in-place mutations where possible
- **Type Safe**: Compile-time dimension checking with `ConstMatrix<T, R, C>` const generics; runtime dimension checking with `Matrix<T>`
- **Comprehensive Test Suite**: 364 tests covering all operations for both `Matrix` and `ConstMatrix`
- **Numerical Stability**: PLU decomposition with partial pivoting for robust computations; closed-form formulas for small matrices (n ≤ 4) for optimal performance

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

// Checked access (no panic)
let safe = a.try_get(0, 1).unwrap();

// Column iteration without allocation
let col: Vec<f64> = a.col_iter(1).copied().collect();
// col() allocates and is deprecated in favor of col_iter().
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

// Matrix exponentiation (positive and negative exponents)
let a_squared = a.pow(2);   // A^2
let a_cubed = a.pow(3);     // A^3
let a_inv = a.pow(-1);      // A^(-1) - matrix inverse
let a_inv_sq = a.pow(-2);   // A^(-2) = (A^(-1))^2
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

### Approximate Equality

```rust
use spectralize::Matrix;

let a = Matrix::new(2, 2, vec![1.0f64, 2.0, 3.0, 4.0]);
let b = Matrix::new(2, 2, vec![1.0f64, 2.0, 3.0, 4.0 + 1e-12]);

// Tolerance-based comparison (exact PartialEq is still available)
assert!(a.approx_eq(&b, 1e-10f64));
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

// Cross product (3D vectors only)
let v1 = Matrix::new(3, 1, vec![1.0, 0.0, 0.0]);  // i
let v2 = Matrix::new(3, 1, vec![0.0, 1.0, 0.0]);  // j
let cross = v1.cross(&v2).unwrap();  // Returns k = (0, 0, 1)

// Transpose
let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let m_t = m.transpose();  // Produces a 3x2 matrix

// Trace (sum of diagonal elements)
let square = Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
let tr = square.trace();  // 1 + 5 + 9 = 15

// Determinant (closed-form for n ≤ 4, PLU decomposition for n > 4)
let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
let det = m.determinant();  // -2.0 (uses fast 2×2 formula: ad - bc)

// Note: Small matrices (n ≤ 4) use optimized closed-form formulas for speed
// Integer determinants use Bareiss elimination for n > 4 (exact but can overflow)
// NaN values are treated as singular (determinant returns 0.0 and is_invertible is false)

// Check if matrix is invertible
let invertible = m.is_invertible();  // true

let singular = Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
let det_singular = singular.determinant();  // 0.0
let invertible_singular = singular.is_invertible();  // false

// Checked variants (no panic on non-square)
let det_checked = m.try_determinant().unwrap();  // Ok(-2.0)
```

### Matrix Inversion

Spectralize provides general matrix inversion for n×n matrices with optimized fast paths for small matrices:

```rust
use spectralize::Matrix;

let a = Matrix::new(2, 2, vec![2.0, 1.0, 1.0, 2.0]);

// Basic inversion
let a_inv = a.inverse();

// Verify: A * A^(-1) = I
let identity = &a * &a_inv;
let expected = Matrix::identity(2, 2);
assert!(identity.approx_eq(&expected, 1e-10));

// Checked inversion (returns Result)
let a_inv_checked = a.try_inverse().unwrap();

// Custom tolerance for near-singular matrices
let nearly_singular = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 1.0 + 1e-10]);

// Strict tolerance: treats it as invertible
let inv1 = nearly_singular.try_inverse_with_tol(1e-12);
assert!(inv1.is_ok());

// Loose tolerance: treats it as singular
let inv2 = nearly_singular.try_inverse_with_tol(1e-8);
assert!(inv2.is_err());

// Singular matrices return errors
let singular = Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
assert!(singular.try_inverse().is_err());  // MatrixError::DimensionMismatch
```

**Available Methods:**
- `inverse()` - Panics if matrix is not square or not invertible
- `try_inverse()` - Returns `Result<Matrix<T>, MatrixError>`
- `inverse_with_tol(tolerance)` - Custom tolerance for floating-point types
- `try_inverse_with_tol(tolerance)` - Checked version with custom tolerance

**Properties Verified:**
- A · A⁻¹ = I (identity)
- A⁻¹ · A = I
- det(A⁻¹) = 1/det(A)
- (A⁻¹)⁻¹ = A

**Type Support:**
- **Floating-point types** (f32, f64, Complex): Full support with tolerance-based singularity detection
- **Integer types**: Not supported (division required for inverse computation)

**Algorithm:**
- **Small matrices (n ≤ 4)**: Uses closed-form adjugate/cofactor formulas for optimal performance with no heap allocations
- **Large matrices (n > 4)**: Uses PLU decomposition with identity-column solving. For each column i of the identity matrix, solves A·x = eᵢ to obtain column i of A⁻¹
- Both approaches are O(n³) but closed-form is significantly faster for small n due to cache efficiency and avoiding general decomposition overhead

### Cross Product

The cross product computes the vector cross product for 3D vectors, producing a vector orthogonal to both inputs following the right-hand rule.

```rust
use spectralize::Matrix;

// Column vectors (3x1)
let i = Matrix::new(3, 1, vec![1.0, 0.0, 0.0]);
let j = Matrix::new(3, 1, vec![0.0, 1.0, 0.0]);
let k = i.cross(&j).unwrap();  // (0, 0, 1)

// Row vectors (1x3) also supported
let v1 = Matrix::new(1, 3, vec![2.0, 3.0, 4.0]);
let v2 = Matrix::new(1, 3, vec![5.0, 6.0, 7.0]);
let result = v1.cross(&v2).unwrap();  // (-3, 6, -3)

// Works with all numeric types
let int_v1 = Matrix::new(3, 1, vec![1i32, 2, 3]);
let int_v2 = Matrix::new(3, 1, vec![4i32, 5, 6]);
let int_cross = int_v1.cross(&int_v2).unwrap();  // (-3, 6, -3)

// Error handling for invalid dimensions
let wrong_size = Matrix::new(2, 1, vec![1.0, 2.0]);
assert!(wrong_size.cross(&i).is_err());  // DimensionMismatch
```

**Properties:**
- **Orthogonality**: The result is perpendicular to both input vectors: `a · (a × b) = 0` and `b · (a × b) = 0`
- **Anti-commutativity**: `a × b = -(b × a)`
- **Right-hand rule**: Follows standard mathematical orientation
- **Parallel vectors**: `a × ka = 0` (zero vector)

**Constraints:**
- Both inputs must be 3-element vectors (3×1 column or 1×3 row)
- Both vectors must have the same shape
- Returns `MatrixError::DimensionMismatch` for invalid inputs

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

// Checked concatenation (no panic)
let with_cols_checked = a.try_with_cols(&b).unwrap();
```

## Compile-Time Dimension Matrices

Spectralize provides `ConstMatrix<T, R, C>` for matrices with compile-time known dimensions. The type system enforces dimension compatibility at compile time, catching dimension mismatches before your code runs!

### Why ConstMatrix?

```rust
use spectralize::matrix::ConstMatrix;

// Dimensions are part of the type
let a: ConstMatrix<f64, 2, 3> = ConstMatrix::new(vec![
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
]);

let b: ConstMatrix<f64, 3, 2> = ConstMatrix::new(vec![
    7.0, 8.0,
    9.0, 10.0,
    11.0, 12.0,
]);

// Matrix multiplication: type system enforces inner dimension match!
let c: ConstMatrix<f64, 2, 2> = a * b;  // OK: (2x3) * (3x2) = (2x2)

// This would be a compile error:
// let wrong: ConstMatrix<f64, 2, 2> = ConstMatrix::identity();
// let result = wrong * a;  // Error: expected ConstMatrix<_, 2, 3>, found ConstMatrix<_, 2, 2>
```

**Key Benefits:**
- **Zero runtime overhead**: No dimension checks at runtime - the type system guarantees correctness
- **Early error detection**: Dimension mismatches caught at compile time, not runtime
- **Better IDE support**: Type hints show exact matrix dimensions
- **Self-documenting code**: Function signatures clearly show expected dimensions

### Basic ConstMatrix Operations

```rust
use spectralize::matrix::ConstMatrix;

// Create matrices with compile-time dimensions
let a: ConstMatrix<f64, 2, 3> = ConstMatrix::new(vec![
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
]);

// Identity matrix - dimensions in the type!
let identity: ConstMatrix<i32, 3, 3> = ConstMatrix::identity();

// Zero matrix
let zero: ConstMatrix<f64, 2, 4> = ConstMatrix::zero();

// Access elements (same API as Matrix)
let element = a.get(0, 1);
let element_ref = a.get_ref(0, 1);  // Zero-cost reference

// Transpose - note how dimensions swap in the type!
let at: ConstMatrix<f64, 3, 2> = a.transpose();
```

### Compile-Time Arithmetic

All arithmetic operations enforce dimension compatibility at compile time:

```rust
use spectralize::matrix::ConstMatrix;

let a: ConstMatrix<i32, 2, 3> = ConstMatrix::new(vec![1, 2, 3, 4, 5, 6]);
let b: ConstMatrix<i32, 2, 3> = ConstMatrix::new(vec![10, 20, 30, 40, 50, 60]);

// Addition/subtraction: requires matching dimensions
let sum = &a + &b;  // OK: both are 2x3
let diff = &a - &b;

// Matrix multiplication: inner dimensions must match
let m1: ConstMatrix<i32, 2, 3> = ConstMatrix::identity();
let m2: ConstMatrix<i32, 3, 2> = ConstMatrix::identity();
let m3: ConstMatrix<i32, 2, 2> = m1 * m2;  // OK: (2x3) * (3x2) = (2x2)

// Scalar multiplication (both directions)
let scaled = &a * 2;
let scaled2 = 3 * &a;

// Works with all types: f32, f64, i32, Complex<f64>, etc.
let complex_m: ConstMatrix<num_complex::Complex<f64>, 2, 2> = ConstMatrix::identity();
```

### Advanced ConstMatrix Operations

```rust
use spectralize::matrix::ConstMatrix;

let m: ConstMatrix<f64, 3, 3> = ConstMatrix::new(vec![
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
    7.0, 8.0, 9.0,
]);

// Trace (sum of diagonal)
let tr = m.trace();  // 1 + 5 + 9 = 15

// Matrix power
let squared = m.pow(2);  // M^2

// Dot product
let v1: ConstMatrix<f64, 1, 3> = ConstMatrix::new(vec![1.0, 2.0, 3.0]);
let v2: ConstMatrix<f64, 1, 3> = ConstMatrix::new(vec![4.0, 5.0, 6.0]);
let dot = v1.dot(&v2);  // 1*4 + 2*5 + 3*6 = 32

// Outer product (returns dynamic Matrix due to const generic limitations)
let v3: ConstMatrix<f64, 3, 1> = ConstMatrix::new(vec![1.0, 2.0, 3.0]);
let v4: ConstMatrix<f64, 2, 1> = ConstMatrix::new(vec![4.0, 5.0]);
let outer = v3.outer(&v4);  // Returns Matrix<f64> (3x2)

// Cross product (3D vectors only)
let i: ConstMatrix<f64, 3, 1> = ConstMatrix::new(vec![1.0, 0.0, 0.0]);
let j: ConstMatrix<f64, 3, 1> = ConstMatrix::new(vec![0.0, 1.0, 0.0]);
let k = i.cross(&j);  // (0, 0, 1)

// Approximate equality with tolerance
let a: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0]);
let b: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0 + 1e-12]);
assert!(a.approx_eq(&b, 1e-10));
```

### ConstMatrix Norms

Matrix norms are available for all ConstMatrix sizes:

```rust
use spectralize::matrix::ConstMatrix;

let m: ConstMatrix<f64, 2, 3> = ConstMatrix::new(vec![
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
]);

// Frobenius norm: sqrt(sum of squares)
let fro = m.frobenius_norm();  // sqrt(1 + 4 + 9 + 16 + 25 + 36)

// Infinity norm: maximum absolute row sum
let inf = m.inf_norm();  // max(|1|+|2|+|3|, |4|+|5|+|6|) = 15

// One norm: maximum absolute column sum
let one = m.one_norm();  // max(|1|+|4|, |2|+|5|, |3|+|6|) = max(5, 7, 9) = 9
```

### Specialized Operations (2x2, 3x3, 4x4)

For small matrices, `ConstMatrix` provides optimized closed-form determinants and inverses:

```rust
use spectralize::matrix::ConstMatrix;

// 2x2 determinant (closed-form: ad - bc)
let m2: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![
    1.0, 2.0,
    3.0, 4.0,
]);
let det2 = m2.determinant();  // -2.0

// 2x2 inverse
let inv2 = m2.inverse().unwrap();  // Returns Option<ConstMatrix<f64, 2, 2>>
let identity = &m2 * &inv2;  // Verify M * M^-1 = I

// 3x3 determinant (cofactor expansion)
let m3: ConstMatrix<i32, 3, 3> = ConstMatrix::new(vec![
    1, 2, 3,
    0, 1, 4,
    5, 6, 0,
]);
let det3 = m3.determinant();

// 3x3 inverse
let m3f: ConstMatrix<f64, 3, 3> = ConstMatrix::identity();
let inv3 = m3f.inverse().unwrap();

// 4x4 determinant (cofactor expansion along first row)
let m4: ConstMatrix<f64, 4, 4> = ConstMatrix::identity();
let det4 = m4.determinant();  // 1.0

// 4x4 inverse (adjugate method)
let inv4 = m4.inverse().unwrap();

// Singular matrices return None
let singular: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![
    1.0, 2.0,
    2.0, 4.0,  // Second row is 2 * first row
]);
assert!(singular.inverse().is_none());
```

**Why specialized operations?**
- **Faster**: Closed-form formulas are faster than general PLU decomposition
- **Type-safe**: Only available for the correct matrix sizes
- **Exact for integers**: No tolerance issues with integer matrices

### Converting Between Matrix and ConstMatrix

```rust
use spectralize::matrix::{Matrix, ConstMatrix};

// ConstMatrix -> Matrix (always succeeds)
let const_m: ConstMatrix<f64, 2, 3> = ConstMatrix::identity();
let dyn_m: Matrix<f64> = const_m.into();

// Matrix -> ConstMatrix (fallible, dimensions must match)
let dyn_m2 = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let const_m2: ConstMatrix<f64, 2, 3> = dyn_m2.try_into().unwrap();

// Dimension mismatch returns Err
let dyn_m3 = Matrix::<f64>::identity(3, 3);
let result: Result<ConstMatrix<f64, 2, 2>, _> = dyn_m3.try_into();
assert!(result.is_err());  // MatrixError::DimensionMismatch
```

**When to use ConstMatrix vs Matrix:**
- **Use ConstMatrix when:**
  - Dimensions are known at compile time
  - You want the type system to catch dimension errors
  - Working with small fixed-size matrices (graphics, physics, etc.)
  - You need specialized 2x2/3x3/4x4 operations

- **Use Matrix when:**
  - Dimensions are only known at runtime
  - Matrix size depends on input data
  - Working with large or variable-sized matrices
  - You need full decomposition features (PLU, etc.)

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
- `pow(n)` - Matrix exponentiation (A^n for n≥0, A^(-n) for invertible matrices)
- `try_pow(n)` - Checked matrix exponentiation (returns Result)
- `dot(other)` - Dot product
- `outer(other)` - Outer product
- `cross(other)` - Cross product (3D vectors only, returns Result)
- `transpose()` - Matrix transpose
- `trace()` - Trace (sum of diagonal elements, square matrices only)
- `determinant()` - Determinant with fast paths: closed-form for n ≤ 4, PLU decomposition for n > 4 (square matrices only)
- `determinant_with_tol(tolerance)` - Determinant with custom singularity tolerance for floating-point types
- `is_invertible()` - Check if matrix is invertible (uses fast paths for n ≤ 4)
- `is_invertible_with_tol(tolerance)` - Invertibility check with custom tolerance for floating-point types
- `inverse()` - Matrix inverse with fast paths: closed-form for n ≤ 4, PLU decomposition for n > 4 (square invertible matrices only)
- `try_inverse()` - Checked matrix inversion (returns Result)
- `inverse_with_tol(tolerance)` - Matrix inverse with custom tolerance for floating-point types
- `try_inverse_with_tol(tolerance)` - Checked matrix inversion with custom tolerance
- `norm_fro()` - Frobenius norm (Euclidean norm): ||A||_F = sqrt(Σ|a_ij|²)
- `norm_one()` - 1-norm (maximum absolute column sum): ||A||_1 = max_j Σ_i |a_ij|
- `norm_inf()` - Infinity norm (maximum absolute row sum): ||A||_∞ = max_i Σ_j |a_ij|

#### Concatenation
- `with_cols(other)` - Horizontal concatenation
- `with_rows(other)` - Vertical concatenation
- `with_row_vec(row)` - Append a row
- `with_col_vec(col)` - Append a column

## Examples

Run the included examples to see various operations in action:

```bash
# Dynamic Matrix examples
cargo run --example float_matrices      # Float operations (f32, f64)
cargo run --example integer_matrices    # Integer operations (i32, i64)
cargo run --example complex_matrices    # Complex number operations
cargo run --example inverse_demo        # Matrix inversion examples
cargo run --example pow_negative_demo   # Negative exponents in pow()

# ConstMatrix examples (compile-time dimensions)
cargo run --example const_matrix_demo        # Basic ConstMatrix operations
cargo run --example const_matrix_arithmetic  # Arithmetic with compile-time checking
```

Each example demonstrates:
- Matrix creation and manipulation
- Arithmetic operations (addition, multiplication, etc.)
- Advanced operations (determinants, norms, cross products, inversion)
- Type-specific features and considerations
- Compile-time dimension safety (ConstMatrix examples)
- Negative exponent support in pow() for invertible matrices

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
├── lib.rs                    # Public API exports
└── matrix/
    ├── mod.rs                # Core Matrix struct and basic operations
    ├── const_matrix.rs       # ConstMatrix with compile-time dimensions (~1150 lines)
    ├── const_matrix_tests.rs # ConstMatrix test suite (105 tests)
    ├── element.rs            # MatrixElement trait and implementations
    ├── arithmetic.rs         # Arithmetic operations (Add, Sub, Mul, cross product, etc.)
    ├── append.rs             # Matrix concatenation operations
    ├── decomposition.rs      # PLU decomposition, determinant, and invertibility
    ├── norm.rs               # Matrix norms (Frobenius, 1-norm, infinity norm)
    └── tests.rs              # Matrix test suite (225 tests)
examples/
├── float_matrices.rs           # Float operations (f32, f64)
├── integer_matrices.rs         # Integer operations (i32, i64)
├── complex_matrices.rs         # Complex number operations
├── const_matrix_demo.rs        # ConstMatrix basics
└── const_matrix_arithmetic.rs  # ConstMatrix arithmetic with type safety
```

## Future Plans

- Additional decompositions:
  - QR decomposition
  - Cholesky decomposition
  - Singular Value Decomposition (SVD)
- Eigenvalue and eigenvector computation
- Linear system solving API (infrastructure exists via PLU decomposition)
- Purpose-built graphics rendering APIs
- Sparse matrix support
- Iterative methods for large systems

## Implementation Notes

### Fast Paths for Small Matrices (n ≤ 4)

For performance-critical applications, Spectralize automatically uses optimized closed-form formulas for determinants and inverses of small matrices:

#### Determinant Fast Paths
- **1×1**: Returns the single element directly
- **2×2**: Uses formula `det = ad - bc` (2 multiplications, 1 subtraction)
- **3×3**: Uses rule of Sarrus with cofactor expansion (9 multiplications, 6 additions/subtractions)
- **4×4**: Uses cofactor expansion along first row, reusing 3×3 helpers (40 multiplications total)

#### Inverse Fast Paths
- **1×1**: Returns reciprocal `1/a` after singularity check
- **2×2**: Uses adjugate formula `(1/det) * [[d, -b], [-c, a]]` (4 multiplications, 4 divisions)
- **3×3**: Computes cofactor matrix and transposes (27 multiplications)
- **4×4**: Computes adjugate via cofactor expansion (reuses 3×3 determinant helpers)

**Performance Characteristics:**
- **Zero heap allocations**: All computation uses stack-allocated arrays and direct indexing
- **No decomposition overhead**: Avoids general PLU setup for matrices where closed-form is faster
- **Cache-friendly**: Unrolled formulas with minimal memory access patterns
- **Numerically accurate**: Direct computation avoids accumulated rounding errors from elimination steps

**Automatic Selection:** The library transparently selects the appropriate algorithm based on matrix size—no API changes required. For matrices larger than 4×4, the implementation falls back to PLU decomposition.

### PLU Decomposition (n > 4)

For large matrices (n > 4), determinant and invertibility checking use an efficient PLU decomposition implementation with partial pivoting:

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

**When PLU is Used:**
- All operations on matrices with dimension n > 4
- Provides robust numerical stability for ill-conditioned matrices
- Supports custom tolerance for floating-point singularity detection

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
