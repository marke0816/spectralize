use super::{Matrix, MatrixElement};
use std::ops::{Add, Mul, Sub};

// Matrix + Matrix (element-wise addition)
impl<T> Add for Matrix<T>
where
    T: MatrixElement + std::fmt::Debug + Add<Output = T>,
{
    type Output = Matrix<T>;

    fn add(mut self, other: Matrix<T>) -> Matrix<T> {
        assert_eq!(self.rows, other.rows, "Row dimensions must match");
        assert_eq!(self.cols, other.cols, "Column dimensions must match");

        // In-place mutation for maximum efficiency
        for i in 0..self.data.len() {
            self.data[i] = self.data[i].clone() + other.data[i].clone();
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

        // In-place mutation, cloning from reference
        for i in 0..self.data.len() {
            self.data[i] = self.data[i].clone() + other.data[i].clone();
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

        // In-place mutation for maximum efficiency
        for i in 0..self.data.len() {
            self.data[i] = self.data[i].clone() - other.data[i].clone();
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

        // In-place mutation, cloning from reference
        for i in 0..self.data.len() {
            self.data[i] = self.data[i].clone() - other.data[i].clone();
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

        // Standard matrix multiplication: (m x n) * (n x p) = (m x p)
        let m = self.rows;
        let n = self.cols;
        let p = other.cols;

        let mut data = Vec::with_capacity(m * p);

        for i in 0..m {
            for j in 0..p {
                let mut sum = T::zero();
                for k in 0..n {
                    sum = sum + (self.data[i * n + k].clone() * other.data[k * p + j].clone());
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

        // Standard matrix multiplication: (m x n) * (n x p) = (m x p)
        let m = self.rows;
        let n = self.cols;
        let p = other.cols;

        let mut data = Vec::with_capacity(m * p);

        for i in 0..m {
            for j in 0..p {
                let mut sum = T::zero();
                for k in 0..n {
                    sum = sum + (self.data[i * n + k].clone() * other.data[k * p + j].clone());
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

        // Standard matrix multiplication: (m x n) * (n x p) = (m x p)
        let m = self.rows;
        let n = self.cols;
        let p = other.cols;

        let mut data = Vec::with_capacity(m * p);

        for i in 0..m {
            for j in 0..p {
                let mut sum = T::zero();
                for k in 0..n {
                    sum = sum + (self.data[i * n + k].clone() * other.data[k * p + j].clone());
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
}
