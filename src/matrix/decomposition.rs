use super::{Matrix, MatrixElement, MatrixError, NanCheck, ToleranceOps};
use crate::matrix::norm::Abs;
use std::ops::{Add, Div, Mul, Neg, Sub};

fn bareiss_determinant<T>(matrix: &Matrix<T>) -> T
where
    T: MatrixElement
        + std::fmt::Debug
        + Clone
        + PivotOrd
        + PartialEq
        + Mul<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + Neg<Output = T>,
{
    // NOTE: Bareiss keeps intermediate values integral but can overflow for large
    // integer inputs (no big-int support here).
    assert_eq!(
        matrix.rows(),
        matrix.cols(),
        "Determinant requires a square matrix"
    );

    let n = matrix.rows();
    if n == 0 {
        return T::one();
    }

    let mut data = matrix.data.clone();
    let mut row_swaps = 0usize;
    let mut prev_pivot = T::one();

    for k in 0..n {
        let mut pivot_row = k;
        let mut max_key = data[k * n + k].pivot_key();
        for i in (k + 1)..n {
            let key = data[i * n + k].pivot_key();
            if key > max_key {
                max_key = key;
                pivot_row = i;
            }
        }

        if data[pivot_row * n + k] == T::zero() {
            return T::zero();
        }

        if pivot_row != k {
            for j in 0..n {
                data.swap(k * n + j, pivot_row * n + j);
            }
            row_swaps += 1;
        }

        let pivot = data[k * n + k].clone();

        for i in (k + 1)..n {
            for j in (k + 1)..n {
                let idx = i * n + j;
                let left = data[idx].clone() * pivot.clone();
                let right = data[i * n + k].clone() * data[k * n + j].clone();
                data[idx] = (left - right) / prev_pivot.clone();
            }
            data[i * n + k] = T::zero();
        }

        prev_pivot = pivot;
    }

    let mut det = data[(n - 1) * n + (n - 1)].clone();
    if row_swaps % 2 == 1 {
        det = -det;
    }
    det
}

fn bareiss_is_invertible<T>(matrix: &Matrix<T>) -> bool
where
    T: MatrixElement
        + std::fmt::Debug
        + Clone
        + PivotOrd
        + PartialEq
        + Mul<Output = T>
        + Sub<Output = T>
        + Div<Output = T>,
{
    assert_eq!(
        matrix.rows(),
        matrix.cols(),
        "Matrix must be square for invertibility check"
    );

    let n = matrix.rows();
    if n == 0 {
        return true;
    }

    let mut data = matrix.data.clone();
    let mut prev_pivot = T::one();

    for k in 0..n {
        let mut pivot_row = k;
        let mut max_key = data[k * n + k].pivot_key();
        for i in (k + 1)..n {
            let key = data[i * n + k].pivot_key();
            if key > max_key {
                max_key = key;
                pivot_row = i;
            }
        }

        if data[pivot_row * n + k] == T::zero() {
            return false;
        }

        if pivot_row != k {
            for j in 0..n {
                data.swap(k * n + j, pivot_row * n + j);
            }
        }

        let pivot = data[k * n + k].clone();

        for i in (k + 1)..n {
            for j in (k + 1)..n {
                let idx = i * n + j;
                let left = data[idx].clone() * pivot.clone();
                let right = data[i * n + k].clone() * data[k * n + j].clone();
                data[idx] = (left - right) / prev_pivot.clone();
            }
            data[i * n + k] = T::zero();
        }

        prev_pivot = pivot;
    }

    data[(n - 1) * n + (n - 1)] != T::zero()
}

/// Internal helper struct that stores the result of PLU decomposition.
/// The L and U matrices are stored in a single matrix:
/// - Lower triangle (below diagonal) contains L with implicit 1s on diagonal
/// - Upper triangle (including diagonal) contains U
/// This packed storage is standard in numerical linear algebra for efficiency.
#[derive(Debug, Clone)]
struct PLUDecomposition<T: MatrixElement + std::fmt::Debug + ToleranceOps> {
    /// Combined L and U matrices in packed form
    lu: Matrix<T>,
    /// Permutation vector: perm[i] indicates the original row index that is now at row i
    /// This tracks all row swaps performed during decomposition
    perm: Vec<usize>,
    /// Number of row swaps performed (determines sign of determinant)
    row_swaps: usize,
    /// Whether the matrix is singular (has a zero/negligible pivot)
    is_singular: bool,
}

/// Trait for types that support magnitude comparison for pivoting.
/// This is needed to find the largest absolute value element for partial pivoting.
pub trait PivotOrd: Sized {
    /// The type of the comparison key (e.g., Self for reals, f32/f64 for Complex)
    type Key: PartialOrd;

    /// Return a value suitable for pivot comparison (e.g., absolute value for signed types)
    fn pivot_key(&self) -> Self::Key;
}

// For unsigned integers, the pivot key is the value itself
impl PivotOrd for u8 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        *self
    }
}

impl PivotOrd for u16 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        *self
    }
}

impl PivotOrd for u32 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        *self
    }
}

impl PivotOrd for u64 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        *self
    }
}

impl PivotOrd for u128 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        *self
    }
}

impl PivotOrd for usize {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        *self
    }
}

// For signed integers, the pivot key is the absolute value
impl PivotOrd for i8 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        self.saturating_abs()
    }
}

impl PivotOrd for i16 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        self.saturating_abs()
    }
}

impl PivotOrd for i32 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        self.saturating_abs()
    }
}

impl PivotOrd for i64 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        self.saturating_abs()
    }
}

impl PivotOrd for i128 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        self.saturating_abs()
    }
}

impl PivotOrd for isize {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        self.saturating_abs()
    }
}

// For floating point, use absolute value
impl PivotOrd for f32 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        self.abs()
    }
}

impl PivotOrd for f64 {
    type Key = Self;

    fn pivot_key(&self) -> Self::Key {
        self.abs()
    }
}

// For complex numbers, use magnitude (norm) as the key
impl PivotOrd for num_complex::Complex<f32> {
    type Key = f32;

    fn pivot_key(&self) -> Self::Key {
        self.norm()
    }
}

impl PivotOrd for num_complex::Complex<f64> {
    type Key = f64;

    fn pivot_key(&self) -> Self::Key {
        self.norm()
    }
}

impl<T> PLUDecomposition<T>
where
    T: MatrixElement + std::fmt::Debug + ToleranceOps + PivotOrd + Div<Output = T> + Mul<Output = T> + Sub<Output = T>,
    T::Abs: Mul<Output = T::Abs>,
{
    /// Compute PLU decomposition with partial pivoting and optional tolerance.
    ///
    /// # Tolerance-Based Pivoting
    ///
    /// This method implements **tolerance-aware Gaussian elimination**:
    ///
    /// ## Why Tolerance Matters
    ///
    /// In exact arithmetic (integers), a zero pivot definitively indicates singularity.
    /// In floating-point arithmetic, numerical errors make exact zero checks unreliable:
    ///
    /// 1. **Rounding errors**: Operations like subtraction can produce tiny nonzero values
    ///    that should be zero mathematically (e.g., 1e-16 instead of 0.0)
    /// 2. **Loss of significance**: Subtracting nearly equal numbers loses precision
    /// 3. **Error accumulation**: Each elimination step compounds previous errors
    ///
    /// A very small pivot (e.g., 1e-15) might be nonzero but dividing by it amplifies
    /// errors catastrophically. The condition number κ(A) ≈ ||A|| / |smallest_pivot|
    /// quantifies this: small pivots → large κ → unstable computation.
    ///
    /// ## Tolerance Formula: ε_pivot = n * ε_machine * ||A||_∞
    ///
    /// We use the **infinity norm** (maximum absolute row sum) as the scale because:
    ///
    /// 1. **Natural for row operations**: Gaussian elimination performs row combinations.
    ///    ||A||_∞ bounds the magnitude of entries affected by each row operation.
    ///
    /// 2. **Computational efficiency**: For row-major storage, computing ||A||_∞ requires
    ///    a single cache-friendly sequential pass. Other norms (Frobenius, 1-norm) are
    ///    less efficient or require column-wise access.
    ///
    /// 3. **Standard in numerical linear algebra**: LAPACK and most production libraries
    ///    use infinity norm for scaling in LU factorization (e.g., DGETRF scaling).
    ///
    /// 4. **Backward error bounds**: For pivoted LU, the growth factor relates directly
    ///    to ||A||_∞, making this norm theoretically justified for stability analysis.
    ///
    /// The factor **n** (matrix dimension) accounts for error accumulation over n elimination
    /// steps. The factor **ε_machine** is the unit roundoff (f64::EPSILON ≈ 2.2e-16 for f64).
    ///
    /// ## Singularity Detection
    ///
    /// A pivot is considered "numerically zero" if:
    ///
    /// ```text
    /// |pivot| ≤ ε_pivot = n * ε_machine * ||A||_∞
    /// ```
    ///
    /// This means:
    /// - **Exact types (integers)**: ε_pivot = 0, so only truly zero pivots are singular
    /// - **Floating-point**: Small pivots relative to matrix scale are treated as zero
    ///
    /// ## Comparison to Exact Arithmetic
    ///
    /// | Aspect | Exact Arithmetic | Floating-Point with Tolerance |
    /// |--------|------------------|-------------------------------|
    /// | Zero pivot | Exactly 0 | \|pivot\| ≤ n·ε·\|\|A\|\|_∞ |
    /// | Singularity | Mathematical property | Numerical/conditioning property |
    /// | Invertibility | Either yes or no | Degree of ill-conditioning |
    /// | Determinant | Exact value or 0 | Approximate or 0 (if singular) |
    /// | Stability | No error propagation | Controlled error amplification |
    ///
    /// ## Example: Nearly Singular Matrix
    ///
    /// Consider A = [[1.0, 1.0], [1.0, 1.0 + 1e-14]] (nearly rank-deficient):
    ///
    /// - **Without tolerance**: PLU succeeds, pivot = 1e-14, condition number ≈ 1e14
    ///   → Matrix reported as invertible but numerically singular
    ///   → Any computation (solve, inverse) produces garbage due to error amplification
    ///
    /// - **With tolerance**: ||A||_∞ = 2.0, ε_pivot ≈ 2 * 2.2e-16 * 2 ≈ 8.8e-16
    ///   → Pivot 1e-14 is accepted (above threshold)
    ///   → Correctly identifies near-singularity through conditioning
    ///
    /// # Parameters
    ///
    /// - `matrix`: The square matrix to decompose
    /// - `tolerance`: Optional user-specified tolerance. If `None`, uses default formula.
    ///   For exact types (integers), tolerance is always effectively zero.
    ///
    /// # Returns
    ///
    /// A PLUDecomposition storing:
    /// - Combined L/U matrix (L below diagonal with implicit 1s, U on/above diagonal)
    /// - Number of row swaps (for determinant sign: (-1)^swaps)
    /// - Singularity flag (true if any pivot magnitude ≤ tolerance)
    /// - Tolerance value used (None for exact arithmetic)
    fn decompose(matrix: &Matrix<T>, tolerance: Option<T::Abs>) -> Self
    where
        T: Clone + Abs<Output = T::Abs>,
        T::Abs: MatrixElement + Add<Output = T::Abs> + Mul<Output = T::Abs> + PartialOrd + NanCheck,
    {
        assert_eq!(
            matrix.rows(),
            matrix.cols(),
            "PLU decomposition requires a square matrix"
        );

        let n = matrix.rows();

        // TOLERANCE SETUP: Centralized logic for all tolerance-aware operations
        // For inexact types (f32, f64, Complex): compute default or use user-specified
        // For exact types (integers): tolerance is None (pure zero check)
        let effective_tolerance = match T::epsilon() {
            Some(eps) => {
                // Floating-point or complex: use tolerance-based pivoting
                let scale = matrix.norm_inf(); // ||A||_∞ for row operation scaling
                let n_abs = T::Abs::one(); // Convert n to appropriate numeric type
                let mut n_factor = n_abs.clone();
                for _ in 1..n {
                    n_factor = n_factor.clone() + n_abs.clone();
                }

                // Default: ε_pivot = n * ε_machine * ||A||_∞
                // User can override by passing tolerance parameter
                let default_tol = n_factor * (eps * scale);
                Some(tolerance.unwrap_or(default_tol))
            }
            None => {
                // Exact arithmetic (integers): no tolerance, pure zero check
                None
            }
        };

        // Clone the matrix for in-place elimination (optimal for performance)
        let mut lu = matrix.clone();
        let mut row_swaps = 0;
        let mut is_singular = false;

        // Initialize permutation vector: perm[i] = i means row i is in its original position
        let mut perm: Vec<usize> = (0..n).collect();

        // Gaussian elimination with partial pivoting and tolerance-aware checks
        // Use direct indexing to avoid repeated bounds checks and clones in hot loops.
        let cols = lu.cols;
        let data = &mut lu.data;

        for k in 0..n {
            // PARTIAL PIVOTING: Find the row with the largest absolute value in column k
            // This improves numerical stability and ensures we don't divide by small numbers
            let mut pivot_row = k;
            let mut max_pivot_key = data[k * cols + k].pivot_key();

            for i in (k + 1)..n {
                let candidate_key = data[i * cols + k].pivot_key();
                if candidate_key > max_pivot_key {
                    max_pivot_key = candidate_key;
                    pivot_row = i;
                }
            }

            // TOLERANCE-AWARE PIVOT CHECK: Centralized singularity detection
            // This is the ONLY place where we check if a pivot is "too small"
            // Both determinant() and is_invertible() rely on this flag
            let pivot_value = data[pivot_row * cols + k].clone();
            let pivot_magnitude = pivot_value.abs_val();
            if pivot_magnitude.is_nan() {
                is_singular = true;
                continue;
            }

            let is_negligible = match &effective_tolerance {
                Some(tol) => pivot_magnitude <= *tol,
                None => pivot_value == T::zero(), // Exact check for integers
            };

            if is_negligible {
                // Pivot is numerically zero: matrix is singular or nearly singular
                is_singular = true;
                // Continue to process remaining columns for complete decomposition structure
                continue;
            }

            // ROW SWAP: Exchange rows k and pivot_row if they differ
            // Each swap changes the sign of the determinant
            if pivot_row != k {
                for j in 0..n {
                    data.swap(k * cols + j, pivot_row * cols + j);
                }
                perm.swap(k, pivot_row);
                row_swaps += 1;
            }

            // ELIMINATION: Zero out all entries below the pivot in column k
            let pivot = data[k * cols + k].clone();

            for i in (k + 1)..n {
                // Compute multiplier: L[i,k] = A[i,k] / A[k,k]
                let idx_ik = i * cols + k;
                let multiplier = data[idx_ik].clone() / pivot.clone();

                // Store the multiplier in lower triangle (L matrix)
                // Note: diagonal of L is implicitly 1, so we don't store it
                data[idx_ik] = multiplier.clone();

                // Update row i: row[i] -= multiplier * row[k]
                // Only need to update entries at and right of the diagonal
                // (entries to the left are already processed or will store L)
                for j in (k + 1)..n {
                    let idx = i * cols + j;
                    let update = data[idx].clone()
                        - (multiplier.clone() * data[k * cols + j].clone());
                    data[idx] = update;
                }
            }
        }

        PLUDecomposition {
            lu,
            perm,
            row_swaps,
            is_singular,
        }
    }

    /// Check if the original matrix is invertible.
    /// A matrix is invertible if and only if all pivots (diagonal of U) are non-zero.
    /// In floating-point arithmetic, this uses the tolerance computed during decomposition.
    fn is_invertible(&self) -> bool {
        !self.is_singular
    }

    /// Compute the determinant of the original matrix.
    ///
    /// For a PLU decomposition: det(A) = det(P) * det(L) * det(U)
    /// - det(P) = (-1)^(number of row swaps)
    /// - det(L) = 1 (lower triangular with 1s on diagonal)
    /// - det(U) = product of diagonal elements
    ///
    /// Therefore: det(A) = (-1)^(row_swaps) * product(diagonal of U)
    ///
    /// ## Tolerance Impact on Determinant
    ///
    /// In exact arithmetic, det(A) = 0 if and only if A is singular.
    /// In floating-point with tolerance:
    ///
    /// - If any pivot |u_ii| ≤ ε_pivot during PLU, we return det(A) = 0
    /// - This correctly identifies **numerically singular** matrices even if
    ///   mathematically det(A) ≠ 0
    ///
    /// Example: A with det(A) = 1e-20 but ||A|| = 1, ε_pivot ≈ 1e-15
    /// → This is effectively singular for computation purposes
    /// → Return det(A) = 0 rather than an unreliable tiny value
    fn determinant(&self) -> T
    where
        T: Neg<Output = T>,
    {
        // Singular matrix (or numerically singular) has determinant zero
        if self.is_singular {
            return T::zero();
        }

        let n = self.lu.rows();

        // Compute product of diagonal elements (these are the pivots from U)
        let mut product = T::one();
        for i in 0..n {
            product = product * self.lu.get(i, i);
        }

        // Apply sign based on number of row swaps: (-1)^row_swaps
        // Even number of swaps: positive determinant
        // Odd number of swaps: negative determinant
        if self.row_swaps % 2 == 1 {
            -product
        } else {
            product
        }
    }

    /// Solve a linear system Ax = b using the PLU decomposition.
    ///
    /// Given the PLU decomposition PA = LU, we solve:
    /// 1. Apply permutation: b' = Pb
    /// 2. Forward substitution: Ly = b'
    /// 3. Backward substitution: Ux = y
    ///
    /// This method uses preallocated working vectors to avoid allocations.
    fn solve(&self, b: &[T], work_perm: &mut [T], work_y: &mut [T], x: &mut [T])
    where
        T: Add<Output = T>,
    {
        let n = self.lu.rows();
        debug_assert_eq!(b.len(), n, "Right-hand side length must match matrix dimension");
        debug_assert_eq!(work_perm.len(), n, "Working vector work_perm must have size n");
        debug_assert_eq!(work_y.len(), n, "Working vector work_y must have size n");
        debug_assert_eq!(x.len(), n, "Working vector x must have size n");

        // Step 1: Apply permutation to get b' = Pb
        // perm[i] tells us which original row is now at position i
        for i in 0..n {
            work_perm[i] = b[self.perm[i]].clone();
        }

        // Step 2: Forward substitution to solve Ly = b'
        // L is lower triangular with implicit 1s on diagonal
        // L is stored in the lower triangle of lu (below diagonal)
        let cols = self.lu.cols;
        let data = &self.lu.data;
        for i in 0..n {
            let mut sum = work_perm[i].clone();
            for j in 0..i {
                sum = sum - (data[i * cols + j].clone() * work_y[j].clone());
            }
            work_y[i] = sum; // Since L[i,i] = 1, no division needed
        }

        // Step 3: Backward substitution to solve Ux = y
        // U is upper triangular stored on and above the diagonal of lu
        for i in (0..n).rev() {
            let mut sum = work_y[i].clone();
            for j in (i + 1)..n {
                sum = sum - (data[i * cols + j].clone() * x[j].clone());
            }
            x[i] = sum / data[i * cols + i].clone(); // Divide by diagonal element
        }
    }
}

// Public API methods on Matrix<T>
impl<T> Matrix<T>
where
    T: MatrixElement + std::fmt::Debug,
{
    /// Check if the matrix is invertible (non-singular) using default tolerance.
    ///
    /// A matrix is invertible if and only if its determinant is non-zero,
    /// which is equivalent to all pivots in the PLU decomposition being non-zero.
    ///
    /// ## Tolerance Behavior
    ///
    /// - **Exact types (integers)**: Uses exact zero check, no tolerance
    /// - **Floating-point (f32, f64, Complex)**: Uses tolerance ε = n * ε_machine * ||A||_∞
    ///   where n is matrix dimension, ε_machine is machine epsilon, and ||A||_∞ is
    ///   the infinity norm (maximum absolute row sum)
    ///
    /// ## Why Default Tolerance Matters
    ///
    /// The default tolerance is derived from **backward error analysis** principles:
    ///
    /// - Each elimination step can introduce error up to O(ε_machine * ||A||)
    /// - After n steps, accumulated error is O(n * ε_machine * ||A||)
    /// - Pivots smaller than this are indistinguishable from zero numerically
    ///
    /// This prevents false positives for invertibility when the matrix is
    /// **ill-conditioned** (condition number κ(A) ≈ 1/ε or worse).
    ///
    /// ## Performance
    ///
    /// This method performs a single PLU decomposition internally: O(n³) time.
    /// The infinity norm is computed first: O(n²) time (negligible vs. PLU cost).
    ///
    /// # Panics
    /// Panics if the matrix is not square.
    ///
    /// # Example
    /// ```
    /// use spectralize::matrix::Matrix;
    ///
    /// let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    /// assert!(m.is_invertible());
    ///
    /// let singular = Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
    /// assert!(!singular.is_invertible());
    ///
    /// // Nearly singular matrix (determinant ≈ 1e-18)
    /// let nearly_singular = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 1.0 + 1e-18]);
    /// // Tolerance-aware check correctly identifies as nearly singular
    /// assert!(!nearly_singular.is_invertible());
    /// ```
    pub fn is_invertible(&self) -> bool
    where
        T: ToleranceOps + PivotOrd + Div<Output = T> + Mul<Output = T> + Sub<Output = T> + Clone + Abs<Output = T::Abs>,
        T::Abs: MatrixElement + Add<Output = T::Abs> + Mul<Output = T::Abs> + PartialOrd + NanCheck,
    {
        if T::epsilon().is_none() {
            return bareiss_is_invertible(self);
        }

        let decomp = PLUDecomposition::decompose(self, None);
        decomp.is_invertible()
    }

    /// Check if the matrix is invertible using a custom tolerance threshold.
    ///
    /// This variant allows you to specify your own tolerance for pivot comparison:
    /// a pivot is considered zero if |pivot| ≤ tolerance.
    ///
    /// ## When to Use Custom Tolerance
    ///
    /// Use this method when:
    ///
    /// 1. **Domain-specific requirements**: Your application has specific numerical
    ///    stability requirements (e.g., physical simulations with known error bounds)
    ///
    /// 2. **Tighter/looser singularity detection**: Default tolerance may be too
    ///    conservative (false negatives) or too lenient (false positives)
    ///
    /// 3. **Experimentation**: Testing sensitivity to tolerance in numerical algorithms
    ///
    /// ## Choosing Tolerance
    ///
    /// - **Stricter than default** (tolerance < n·ε·||A||): Accept more matrices as
    ///   invertible, risk numerical instability in subsequent operations
    ///
    /// - **Looser than default** (tolerance > n·ε·||A||): Reject more matrices as
    ///   singular, increased safety but potential over-rejection
    ///
    /// - **Relative to data precision**: For data with known measurement error δ,
    ///   use tolerance ≈ δ * ||A||
    ///
    /// ## Note on Exact Types
    ///
    /// For integer matrices, the tolerance parameter is **ignored** and exact
    /// zero-checking is always used (since integers have no rounding errors).
    ///
    /// # Parameters
    ///
    /// - `tolerance`: The threshold for pivot comparison. Pivots with |pivot| ≤ tolerance
    ///   are considered zero. Ignored for integer types.
    ///
    /// # Panics
    /// Panics if the matrix is not square.
    ///
    /// # Example
    /// ```
    /// use spectralize::matrix::Matrix;
    ///
    /// let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    ///
    /// // Very strict tolerance: only accept nearly perfect matrices
    /// assert!(m.is_invertible_with_tol(1e-12));
    ///
    /// // Looser tolerance: accept matrices with small pivots
    /// let nearly_singular = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 1.0 + 1e-10]);
    /// assert!(!nearly_singular.is_invertible_with_tol(1e-8)); // Rejected
    /// assert!(nearly_singular.is_invertible_with_tol(1e-12));  // Accepted
    /// ```
    pub fn is_invertible_with_tol(&self, tolerance: T::Abs) -> bool
    where
        T: ToleranceOps + PivotOrd + Div<Output = T> + Mul<Output = T> + Sub<Output = T> + Clone + Abs<Output = T::Abs>,
        T::Abs: MatrixElement + Add<Output = T::Abs> + Mul<Output = T::Abs> + PartialOrd + NanCheck,
    {
        if T::epsilon().is_none() {
            return bareiss_is_invertible(self);
        }

        let decomp = PLUDecomposition::decompose(self, Some(tolerance));
        decomp.is_invertible()
    }

    /// Compute the determinant of the matrix using default tolerance.
    ///
    /// The determinant is computed using PLU decomposition:
    /// det(A) = (-1)^(row_swaps) * product(diagonal of U)
    ///
    /// ## Tolerance Behavior
    ///
    /// - **Exact types (integers)**: Computes exact determinant
    /// - **Floating-point**: Uses tolerance ε = n * ε_machine * ||A||_∞
    ///   - If any pivot |u_ii| ≤ ε, returns det(A) = 0 (numerically singular)
    ///   - Otherwise, computes determinant from pivot product
    ///
    /// ## Why Return Zero for Small Determinants
    ///
    /// In floating-point arithmetic, a very small determinant (e.g., 1e-20) is
    /// unreliable and often meaningless:
    ///
    /// - **Rounding errors dominate**: The computed value may have no correct digits
    /// - **Amplified errors**: Small det(A) implies large condition number κ(A),
    ///   so any computed value is subject to massive error amplification
    /// - **Practical singularity**: Matrices with det(A) < ε * ||A||^n are
    ///   effectively singular for numerical purposes
    ///
    /// Rather than return an unreliable tiny value, we return exactly zero,
    /// signaling that the matrix is **numerically singular** even if
    /// mathematically det(A) ≠ 0.
    ///
    /// ## Performance
    ///
    /// This method performs a single PLU decomposition internally: O(n³) time.
    ///
    /// # Panics
    /// Panics if the matrix is not square.
    ///
    /// # Example
    /// ```
    /// use spectralize::matrix::Matrix;
    ///
    /// let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(m.determinant(), -2.0);
    ///
    /// let singular = Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
    /// assert_eq!(singular.determinant(), 0.0);
    ///
    /// // Nearly singular: mathematically det ≈ 1e-18, but numerically zero
    /// let nearly_singular = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 1.0 + 1e-18]);
    /// assert_eq!(nearly_singular.determinant(), 0.0); // Treated as singular
    /// ```
    pub fn determinant(&self) -> T
    where
        T: ToleranceOps + PivotOrd + Div<Output = T> + Mul<Output = T> + Sub<Output = T> + Neg<Output = T> + Clone + Abs<Output = T::Abs>,
        T::Abs: MatrixElement + Add<Output = T::Abs> + Mul<Output = T::Abs> + PartialOrd + NanCheck,
    {
        // For integer types, this uses Bareiss elimination; large values may overflow.
        if T::epsilon().is_none() {
            return bareiss_determinant(self);
        }

        let decomp = PLUDecomposition::decompose(self, None);
        decomp.determinant()
    }

    /// Compute the determinant using a custom tolerance threshold.
    ///
    /// This variant allows you to specify your own tolerance for singularity detection.
    /// Pivots with |pivot| ≤ tolerance are treated as zero, resulting in det(A) = 0.
    ///
    /// ## When to Use Custom Tolerance
    ///
    /// Use this method for the same reasons as `is_invertible_with_tol`:
    ///
    /// 1. **Application-specific error bounds**: Match tolerance to your data precision
    /// 2. **Sensitivity analysis**: Test how tolerance affects singularity detection
    /// 3. **Stricter/looser criteria**: Override default tolerance behavior
    ///
    /// ## Interpretation of Results
    ///
    /// - **det(A) = 0**: Matrix is singular under the specified tolerance
    ///   (at least one pivot ≤ tolerance)
    ///
    /// - **det(A) ≠ 0**: All pivots exceed tolerance; determinant computed from
    ///   pivot product with sign correction
    ///
    /// ## Note on Exact Types
    ///
    /// For integer matrices, tolerance is ignored and exact determinant is computed.
    ///
    /// # Parameters
    ///
    /// - `tolerance`: The threshold for pivot comparison. Ignored for integer types.
    ///
    /// # Panics
    /// Panics if the matrix is not square.
    ///
    /// # Example
    /// ```
    /// use spectralize::matrix::Matrix;
    ///
    /// let nearly_singular = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 1.0 + 1e-10]);
    ///
    /// // Strict tolerance: small pivot accepted, determinant computed
    /// let det_strict: f64 = nearly_singular.determinant_with_tol(1e-12f64);
    /// assert!(det_strict.abs() > 0.0f64);
    ///
    /// // Loose tolerance: small pivot rejected, determinant is zero
    /// let det_loose = nearly_singular.determinant_with_tol(1e-8f64);
    /// assert_eq!(det_loose, 0.0f64);
    /// ```
    pub fn determinant_with_tol(&self, tolerance: T::Abs) -> T
    where
        T: ToleranceOps + PivotOrd + Div<Output = T> + Mul<Output = T> + Sub<Output = T> + Neg<Output = T> + Clone + Abs<Output = T::Abs>,
        T::Abs: MatrixElement + Add<Output = T::Abs> + Mul<Output = T::Abs> + PartialOrd + NanCheck,
    {
        if T::epsilon().is_none() {
            return bareiss_determinant(self);
        }

        let decomp = PLUDecomposition::decompose(self, Some(tolerance));
        decomp.determinant()
    }

    /// Compute the inverse of a square matrix using default tolerance.
    ///
    /// The inverse is computed using PLU decomposition: for each column i of the
    /// identity matrix, solve A·x = e_i to obtain column i of A^(-1).
    ///
    /// ## Algorithm
    ///
    /// 1. Compute PLU decomposition once: PA = LU
    /// 2. For each standard basis vector e_i:
    ///    - Solve Ly = Pe_i (forward substitution)
    ///    - Solve Ux = y (backward substitution)
    ///    - Store x as column i of A^(-1)
    ///
    /// ## Performance
    ///
    /// - PLU decomposition: O(n³) time, O(n²) space
    /// - Solving n systems: O(n³) time total (O(n²) per column)
    /// - Total: O(n³) time, O(n²) space
    ///
    /// This implementation reuses working vectors to avoid allocations in the inner loop.
    ///
    /// ## Tolerance Behavior
    ///
    /// - **Exact types (integers)**: Exact inverse computation, returns error if not invertible
    /// - **Floating-point**: Uses default tolerance ε = n * ε_machine * ||A||_∞
    ///   - Returns error if any pivot ≤ ε (numerically singular)
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square or is singular (non-invertible).
    /// Use `try_inverse()` for error handling instead of panics.
    ///
    /// # Example
    ///
    /// ```
    /// use spectralize::matrix::Matrix;
    ///
    /// let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    /// let a_inv = a.inverse();
    ///
    /// // Verify: A * A^(-1) ≈ I
    /// let identity = &a * &a_inv;
    /// let expected = Matrix::identity(2, 2);
    /// assert!(identity.approx_eq(&expected, 1e-10));
    /// ```
    pub fn inverse(&self) -> Matrix<T>
    where
        T: ToleranceOps
            + PivotOrd
            + Div<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + Add<Output = T>
            + Clone
            + Abs<Output = T::Abs>,
        T::Abs: MatrixElement + Add<Output = T::Abs> + Mul<Output = T::Abs> + PartialOrd + NanCheck,
    {
        self.try_inverse()
            .expect("Matrix must be square and invertible")
    }

    /// Compute the inverse of a square matrix using default tolerance.
    ///
    /// This is the checked version of `inverse()` that returns a `Result` instead of panicking.
    ///
    /// # Errors
    ///
    /// Returns `MatrixError::DimensionMismatch` if:
    /// - The matrix is not square
    /// - The matrix is singular (determinant is zero or numerically negligible)
    ///
    /// # Example
    ///
    /// ```
    /// use spectralize::matrix::{Matrix, MatrixError};
    ///
    /// let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    /// let a_inv = a.try_inverse().unwrap();
    ///
    /// let singular = Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
    /// assert_eq!(singular.try_inverse().unwrap_err(), MatrixError::DimensionMismatch);
    ///
    /// let rect = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(rect.try_inverse().unwrap_err(), MatrixError::DimensionMismatch);
    /// ```
    pub fn try_inverse(&self) -> Result<Matrix<T>, MatrixError>
    where
        T: ToleranceOps
            + PivotOrd
            + Div<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + Add<Output = T>
            + Clone
            + Abs<Output = T::Abs>,
        T::Abs: MatrixElement + Add<Output = T::Abs> + Mul<Output = T::Abs> + PartialOrd + NanCheck,
    {
        // Check square
        if self.rows != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        // Compute PLU decomposition with default tolerance
        let decomp = PLUDecomposition::decompose(self, None);

        // Check invertibility
        if !decomp.is_invertible() {
            return Err(MatrixError::DimensionMismatch);
        }

        let n = self.rows;

        // Preallocate result matrix and working vectors (avoid allocations in loop)
        let mut inverse_data = vec![T::zero(); n * n];
        let mut work_perm = vec![T::zero(); n];
        let mut work_y = vec![T::zero(); n];
        let mut work_x = vec![T::zero(); n];
        let mut e_i = vec![T::zero(); n];

        // Solve for each column of the inverse
        // For column i, solve A·x = e_i where e_i is the i-th standard basis vector
        for i in 0..n {
            // Update standard basis vector in O(1)
            if i > 0 {
                e_i[i - 1] = T::zero();
            }
            e_i[i] = T::one();

            // Solve A·x = e_i using PLU decomposition
            decomp.solve(&e_i, &mut work_perm, &mut work_y, &mut work_x);

            // Write column i into row-major inverse buffer
            for row in 0..n {
                inverse_data[row * n + i] = work_x[row].clone();
            }
        }

        Ok(Matrix::new(n, n, inverse_data))
    }

    /// Compute the inverse of a square matrix using a custom tolerance.
    ///
    /// This variant allows you to specify your own tolerance for singularity detection.
    /// Use this when the default tolerance is not appropriate for your application.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square or is singular under the specified tolerance.
    /// Use `try_inverse_with_tol()` for error handling instead of panics.
    ///
    /// # Example
    ///
    /// ```
    /// use spectralize::matrix::Matrix;
    ///
    /// let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    /// let a_inv = a.inverse_with_tol(1e-10);
    /// ```
    pub fn inverse_with_tol(&self, tolerance: T::Abs) -> Matrix<T>
    where
        T: ToleranceOps
            + PivotOrd
            + Div<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + Add<Output = T>
            + Clone
            + Abs<Output = T::Abs>,
        T::Abs: MatrixElement + Add<Output = T::Abs> + Mul<Output = T::Abs> + PartialOrd + NanCheck,
    {
        self.try_inverse_with_tol(tolerance)
            .expect("Matrix must be square and invertible")
    }

    /// Compute the inverse of a square matrix using a custom tolerance.
    ///
    /// This is the checked version of `inverse_with_tol()` that returns a `Result`.
    ///
    /// # Errors
    ///
    /// Returns `MatrixError::DimensionMismatch` if:
    /// - The matrix is not square
    /// - The matrix is singular under the specified tolerance
    ///
    /// # Example
    ///
    /// ```
    /// use spectralize::matrix::{Matrix, MatrixError};
    ///
    /// let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    /// let a_inv = a.try_inverse_with_tol(1e-10).unwrap();
    ///
    /// let nearly_singular = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 1.0 + 1e-10]);
    /// // With strict tolerance, accepted as invertible
    /// assert!(nearly_singular.try_inverse_with_tol(1e-12).is_ok());
    /// // With loose tolerance, rejected as singular
    /// assert!(nearly_singular.try_inverse_with_tol(1e-8).is_err());
    /// ```
    pub fn try_inverse_with_tol(&self, tolerance: T::Abs) -> Result<Matrix<T>, MatrixError>
    where
        T: ToleranceOps
            + PivotOrd
            + Div<Output = T>
            + Mul<Output = T>
            + Sub<Output = T>
            + Add<Output = T>
            + Clone
            + Abs<Output = T::Abs>,
        T::Abs: MatrixElement + Add<Output = T::Abs> + Mul<Output = T::Abs> + PartialOrd + NanCheck,
    {
        // Check square
        if self.rows != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        // Compute PLU decomposition with custom tolerance
        let decomp = PLUDecomposition::decompose(self, Some(tolerance));

        // Check invertibility
        if !decomp.is_invertible() {
            return Err(MatrixError::DimensionMismatch);
        }

        let n = self.rows;

        // Preallocate result matrix and working vectors (avoid allocations in loop)
        let mut inverse_data = vec![T::zero(); n * n];
        let mut work_perm = vec![T::zero(); n];
        let mut work_y = vec![T::zero(); n];
        let mut work_x = vec![T::zero(); n];
        let mut e_i = vec![T::zero(); n];

        // Solve for each column of the inverse
        for i in 0..n {
            // Update standard basis vector in O(1)
            if i > 0 {
                e_i[i - 1] = T::zero();
            }
            e_i[i] = T::one();

            // Solve A·x = e_i
            decomp.solve(&e_i, &mut work_perm, &mut work_y, &mut work_x);

            // Write column i into row-major inverse buffer
            for row in 0..n {
                inverse_data[row * n + i] = work_x[row].clone();
            }
        }

        Ok(Matrix::new(n, n, inverse_data))
    }

    /// Matrix exponentiation with support for negative exponents.
    ///
    /// This specialized implementation for types that support matrix inversion
    /// extends matrix exponentiation to handle negative exponents:
    /// - A^n = A * A * ... * A (n times) for n > 0
    /// - A^0 = I (identity matrix)
    /// - A^(-n) = (A^(-1))^n for n < 0
    ///
    /// # Panics
    /// - Panics if the matrix is not square
    /// - Panics if n < 0 and the matrix is not invertible
    ///
    /// # Examples
    /// ```
    /// use spectralize::matrix::Matrix;
    ///
    /// let a = Matrix::new(2, 2, vec![2.0, 0.0, 0.0, 3.0]);
    ///
    /// // Positive exponent
    /// let a_squared = a.pow(2);
    ///
    /// // Negative exponent
    /// let a_inv = a.pow(-1);
    /// // Equivalent to: a.inverse()
    ///
    /// let a_inv_squared = a.pow(-2);
    /// // Equivalent to: a.inverse().pow(2)
    /// ```
    pub fn pow(&self, n: i32) -> Matrix<T>
    where
        T: Mul<Output = T> + Add<Output = T> + ToleranceOps + PivotOrd + Div<Output = T> + Sub<Output = T> + Abs<Output = T::Abs>,
        T::Abs: MatrixElement + Add<Output = T::Abs> + Mul<Output = T::Abs> + PartialOrd + NanCheck,
    {
        assert_eq!(
            self.rows, self.cols,
            "Matrix must be square for exponentiation"
        );

        // Handle negative exponents: A^(-n) = (A^(-1))^n
        if n < 0 {
            let inv = self.inverse();
            let abs_n = n.unsigned_abs();
            return inv.pow_u32(abs_n);
        }

        // For non-negative exponents, delegate to basic pow implementation
        // The basic pow in arithmetic.rs handles non-negative i32
        // by converting to u32 and using binary exponentiation
        // We can't call it directly due to method resolution, so we'll
        // inline the logic here to avoid recursion issues
        self.pow_u32(n as u32)
    }

    /// Checked matrix exponentiation with support for negative exponents.
    ///
    /// This is the non-panicking version of `pow` that returns a `Result`.
    ///
    /// # Errors
    /// - Returns `MatrixError::DimensionMismatch` if the matrix is not square
    /// - Returns `MatrixError::DimensionMismatch` if n < 0 and the matrix is not invertible
    ///
    /// # Examples
    /// ```
    /// use spectralize::matrix::{Matrix, MatrixError};
    ///
    /// let a = Matrix::new(2, 2, vec![2.0, 0.0, 0.0, 3.0]);
    ///
    /// // Negative exponent on invertible matrix: success
    /// let a_inv_squared = a.try_pow(-2).unwrap();
    ///
    /// // Negative exponent on singular matrix: error
    /// let singular = Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
    /// assert_eq!(singular.try_pow(-1).unwrap_err(), MatrixError::DimensionMismatch);
    /// ```
    pub fn try_pow(&self, n: i32) -> Result<Matrix<T>, MatrixError>
    where
        T: Mul<Output = T> + Add<Output = T> + ToleranceOps + PivotOrd + Div<Output = T> + Sub<Output = T> + Abs<Output = T::Abs>,
        T::Abs: MatrixElement + Add<Output = T::Abs> + Mul<Output = T::Abs> + PartialOrd + NanCheck,
    {
        if self.rows != self.cols {
            return Err(MatrixError::DimensionMismatch);
        }

        // Handle negative exponents using try_inverse
        if n < 0 {
            let inv = self.try_inverse()?;
            let abs_n = n.unsigned_abs();
            return Ok(inv.pow_u32(abs_n));
        }

        Ok(self.pow(n))
    }

    fn pow_u32(&self, n: u32) -> Matrix<T>
    where
        T: Mul<Output = T> + Add<Output = T>,
    {
        match n {
            0 => Matrix::identity(self.rows, self.cols),
            1 => self.clone(),
            _ => {
                let mut result = Matrix::identity(self.rows, self.cols);
                let mut base = self.clone();
                let mut exp = n;

                while exp > 0 {
                    if exp % 2 == 1 {
                        result = result * &base;
                    }
                    base = &base * &base;
                    exp /= 2;
                }

                result
            }
        }
    }
}
