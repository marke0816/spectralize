use super::{Matrix, MatrixElement};
use std::ops::{Add, Mul, Sub};

fn mul_with_transposed<T>(lhs: &Matrix<T>, rhs_t: &Matrix<T>) -> Matrix<T>
where
    T: MatrixElement + std::fmt::Debug + Mul<Output = T> + Add<Output = T>,
{
    assert_eq!(
        lhs.cols, rhs_t.cols,
        "Matrix dimensions incompatible for multiplication"
    );

    let m = lhs.rows;
    let n = lhs.cols;
    let p = rhs_t.rows;

    let mut data = Vec::with_capacity(m * p);

    for i in 0..m {
        let row = &lhs.data[i * n..i * n + n];
        for j in 0..p {
            let col = &rhs_t.data[j * n..j * n + n];
            let mut sum = T::zero();
            for k in 0..n {
                sum = sum + (row[k].clone() * col[k].clone());
            }
            data.push(sum);
        }
    }

    Matrix {
        rows: m,
        cols: p,
        data,
    }
}

// Matrix + Matrix (element-wise addition)
impl<T> Add for Matrix<T>
where
    T: MatrixElement + std::fmt::Debug + Add<Output = T>,
{
    type Output = Matrix<T>;

    fn add(mut self, other: Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows, "Row dimensions must match");
        assert_eq!(self.cols, other.cols, "Column dimensions must match");

        // In-place mutation without extra zeroing
        for (lhs, rhs) in self.data.iter_mut().zip(other.data.into_iter()) {
            *lhs = lhs.clone() + rhs;
        }

        self
    }
}

// Matrix + &Matrix (element-wise addition with reference)
impl<T> Add<&Matrix<T>> for Matrix<T>
where
    T: MatrixElement + std::fmt::Debug + Add<Output = T>,
{
    type Output = Matrix<T>;

    fn add(mut self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows, "Row dimensions must match");
        assert_eq!(self.cols, other.cols, "Column dimensions must match");

        // In-place mutation without extra zeroing
        for (lhs, rhs) in self.data.iter_mut().zip(other.data.iter()) {
            *lhs = lhs.clone() + rhs.clone();
        }

        self
    }
}

// &Matrix + &Matrix (element-wise addition with references)
impl<T> Add for &Matrix<T>
where
    T: MatrixElement + std::fmt::Debug + Add<Output = T>,
{
    type Output = Matrix<T>;

    fn add(self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows, "Row dimensions must match");
        assert_eq!(self.cols, other.cols, "Column dimensions must match");

        // Both references, must allocate new vector
        let mut data = Vec::with_capacity(self.data.len());
        for i in 0..self.data.len() {
            data.push(self.data[i].clone() + other.data[i].clone());
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

// Matrix - Matrix (element-wise subtraction)
impl<T> Sub for Matrix<T>
where
    T: MatrixElement + std::fmt::Debug + Sub<Output = T>,
{
    type Output = Matrix<T>;

    fn sub(mut self, other: Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows, "Row dimensions must match");
        assert_eq!(self.cols, other.cols, "Column dimensions must match");

        // In-place mutation without extra zeroing
        for (lhs, rhs) in self.data.iter_mut().zip(other.data.into_iter()) {
            *lhs = lhs.clone() - rhs;
        }

        self
    }
}

// Matrix - &Matrix (element-wise subtraction with reference)
impl<T> Sub<&Matrix<T>> for Matrix<T>
where
    T: MatrixElement + std::fmt::Debug + Sub<Output = T>,
{
    type Output = Matrix<T>;

    fn sub(mut self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows, "Row dimensions must match");
        assert_eq!(self.cols, other.cols, "Column dimensions must match");

        // In-place mutation without extra zeroing
        for (lhs, rhs) in self.data.iter_mut().zip(other.data.iter()) {
            *lhs = lhs.clone() - rhs.clone();
        }

        self
    }
}

// &Matrix - &Matrix (element-wise subtraction with references)
impl<T> Sub for &Matrix<T>
where
    T: MatrixElement + std::fmt::Debug + Sub<Output = T>,
{
    type Output = Matrix<T>;

    fn sub(self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows, "Row dimensions must match");
        assert_eq!(self.cols, other.cols, "Column dimensions must match");

        // Both references, must allocate new vector
        let mut data = Vec::with_capacity(self.data.len());
        for i in 0..self.data.len() {
            data.push(self.data[i].clone() - other.data[i].clone());
        }

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

// Matrix * Matrix (matrix multiplication)
impl<T> Mul for Matrix<T>
where
    T: MatrixElement + std::fmt::Debug + Mul<Output = T> + Add<Output = T>,
{
    type Output = Matrix<T>;

    fn mul(self, other: Matrix<T>) -> Matrix<T> {
        assert_eq!(
            self.cols, other.rows,
            "Matrix dimensions incompatible for multiplication"
        );

        let other_t = other.transpose();
        mul_with_transposed(&self, &other_t)
    }
}

// Matrix * &Matrix (matrix multiplication with reference)
impl<T> Mul<&Matrix<T>> for Matrix<T>
where
    T: MatrixElement + std::fmt::Debug + Mul<Output = T> + Add<Output = T>,
{
    type Output = Matrix<T>;

    fn mul(self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(
            self.cols, other.rows,
            "Matrix dimensions incompatible for multiplication"
        );

        let other_t = other.transpose();
        mul_with_transposed(&self, &other_t)
    }
}

// &Matrix * &Matrix (matrix multiplication with references)
impl<T> Mul for &Matrix<T>
where
    T: MatrixElement + std::fmt::Debug + Mul<Output = T> + Add<Output = T>,
{
    type Output = Matrix<T>;

    fn mul(self, other: &Matrix<T>) -> Matrix<T> {
        assert_eq!(
            self.cols, other.rows,
            "Matrix dimensions incompatible for multiplication"
        );

        let other_t = other.transpose();
        mul_with_transposed(self, &other_t)
    }
}

// Scalar multiplication: Matrix * scalar
impl<T> Mul<T> for Matrix<T>
where
    T: MatrixElement + std::fmt::Debug + Mul<Output = T>,
{
    type Output = Matrix<T>;

    fn mul(mut self, scalar: T) -> Matrix<T> {
        // In-place mutation without extra zeroing
        for elem in self.data.iter_mut() {
            *elem = elem.clone() * scalar.clone();
        }
        self
    }
}

// Scalar multiplication: &Matrix * scalar
impl<T> Mul<T> for &Matrix<T>
where
    T: MatrixElement + std::fmt::Debug + Mul<Output = T>,
{
    type Output = Matrix<T>;

    fn mul(self, scalar: T) -> Matrix<T> {
        // Must allocate new vector for reference
        let mut data = Vec::with_capacity(self.data.len());
        for i in 0..self.data.len() {
            data.push(self.data[i].clone() * scalar.clone());
        }
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        }
    }
}

// Scalar multiplication: scalar * Matrix (f64)
impl Mul<Matrix<f64>> for f64 {
    type Output = Matrix<f64>;

    fn mul(self, mut matrix: Matrix<f64>) -> Matrix<f64> {
        for i in 0..matrix.data.len() {
            matrix.data[i] = self * matrix.data[i];
        }
        matrix
    }
}

// Scalar multiplication: scalar * &Matrix (f64)
impl Mul<&Matrix<f64>> for f64 {
    type Output = Matrix<f64>;

    fn mul(self, matrix: &Matrix<f64>) -> Matrix<f64> {
        let mut data = Vec::with_capacity(matrix.data.len());
        for i in 0..matrix.data.len() {
            data.push(self * matrix.data[i]);
        }
        Matrix {
            rows: matrix.rows,
            cols: matrix.cols,
            data,
        }
    }
}

// Scalar multiplication: scalar * Matrix (f32)
impl Mul<Matrix<f32>> for f32 {
    type Output = Matrix<f32>;

    fn mul(self, mut matrix: Matrix<f32>) -> Matrix<f32> {
        for i in 0..matrix.data.len() {
            matrix.data[i] = self * matrix.data[i];
        }
        matrix
    }
}

// Scalar multiplication: scalar * &Matrix (f32)
impl Mul<&Matrix<f32>> for f32 {
    type Output = Matrix<f32>;

    fn mul(self, matrix: &Matrix<f32>) -> Matrix<f32> {
        let mut data = Vec::with_capacity(matrix.data.len());
        for i in 0..matrix.data.len() {
            data.push(self * matrix.data[i]);
        }
        Matrix {
            rows: matrix.rows,
            cols: matrix.cols,
            data,
        }
    }
}

// Scalar multiplication: scalar * Matrix (i32)
impl Mul<Matrix<i32>> for i32 {
    type Output = Matrix<i32>;

    fn mul(self, mut matrix: Matrix<i32>) -> Matrix<i32> {
        for i in 0..matrix.data.len() {
            matrix.data[i] = self * matrix.data[i];
        }
        matrix
    }
}

// Scalar multiplication: scalar * &Matrix (i32)
impl Mul<&Matrix<i32>> for i32 {
    type Output = Matrix<i32>;

    fn mul(self, matrix: &Matrix<i32>) -> Matrix<i32> {
        let mut data = Vec::with_capacity(matrix.data.len());
        for i in 0..matrix.data.len() {
            data.push(self * matrix.data[i]);
        }
        Matrix {
            rows: matrix.rows,
            cols: matrix.cols,
            data,
        }
    }
}

// Additional matrix operations
impl<T> Matrix<T>
where
    T: MatrixElement + std::fmt::Debug,
{
    /// Matrix exponentiation: raises a square matrix to a non-negative integer power
    /// Uses binary exponentiation for O(log n) complexity
    /// A^n = A * A * ... * A (n times)
    /// A^0 = I (identity matrix)
    /// A^1 = A
    pub fn pow(&self, n: u32) -> Matrix<T>
    where
        T: Mul<Output = T> + Add<Output = T>,
    {
        assert_eq!(
            self.rows, self.cols,
            "Matrix must be square for exponentiation"
        );

        // TODO: Add support for negative exponents when matrix inversion is implemented
        // For A^(-n), we would need to compute (A^(-1))^n

        match n {
            0 => {
                // A^0 = Identity matrix
                let mut data = vec![T::zero(); self.rows * self.rows];
                for i in 0..self.rows {
                    data[i * self.rows + i] = T::one();
                }
                Matrix {
                    rows: self.rows,
                    cols: self.cols,
                    data,
                }
            }
            1 => {
                // A^1 = A
                self.clone()
            }
            _ => {
                // Binary exponentiation: O(log n) instead of O(n)
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

    /// Dot product (inner product) of two matrices treated as vectors
    /// Computes the sum of element-wise products: sum(a[i] * b[i])
    /// Both matrices must have the same total number of elements
    /// Returns a scalar value
    pub fn dot(&self, other: &Matrix<T>) -> T
    where
        T: Mul<Output = T> + Add<Output = T>,
    {
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "Matrices must have the same total number of elements for dot product"
        );

        let mut result = T::zero();
        for i in 0..self.data.len() {
            result = result + (self.data[i].clone() * other.data[i].clone());
        }
        result
    }

    /// Outer product of two matrices treated as vectors
    /// For vector u (length m) and vector v (length n), produces an m×n matrix
    /// where result[i,j] = u[i] * v[j]
    pub fn outer(&self, other: &Matrix<T>) -> Matrix<T>
    where
        T: Mul<Output = T>,
    {
        let m = self.data.len();
        let n = other.data.len();

        let mut data = Vec::with_capacity(m * n);

        for i in 0..m {
            for j in 0..n {
                data.push(self.data[i].clone() * other.data[j].clone());
            }
        }

        Matrix {
            rows: m,
            cols: n,
            data,
        }
    }
}
