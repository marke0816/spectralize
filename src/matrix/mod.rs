pub mod append;
pub mod element;

pub use element::MatrixElement;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T: MatrixElement + std::fmt::Debug> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: MatrixElement + std::fmt::Debug> Matrix<T> {
    /// Create a new matrix from a Vec<T>
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(rows * cols, data.len());
        Self { rows, cols, data }
    }

    /// Create a rows x cols zero matrix
    pub fn zero(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![T::zero(); rows * cols],
        }
    }

    /// Create a rows x cols identity matrix
    pub fn identity(rows: usize, cols: usize) -> Self {
        let mut data = vec![T::zero(); rows * cols];

        (0..rows.min(cols)).for_each(|i| data[i * cols + i] = T::one());

        Self { rows, cols, data }
    }

    /// Create a permutation matrix from a vector of column indices.
    /// The `perm` vector should contain integers 1..=cols, one for each row
    pub fn perm(rows: usize, cols: usize, perm: Vec<usize>) -> Self {
        assert_eq!(
            rows,
            perm.len(),
            "Length of permutation vector must match rows"
        );
        let mut data = vec![T::zero(); rows * cols];

        for (row, &col_index) in perm.iter().enumerate() {
            assert!(
                col_index >= 1 && col_index <= cols,
                "Column indices must be 1-based and <= cols"
            );
            data[row * cols + (col_index - 1)] = T::one();
        }

        Self { rows, cols, data }
    }

    /// Get the number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Safe getter
    pub fn get(&self, row: usize, col: usize) -> T {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        self.data[row * self.cols + col].clone()
    }

    /// Safe setter
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        self.data[row * self.cols + col] = value;
    }

    /// Iterator over a row
    pub fn row(&self, row: usize) -> &[T] {
        assert!(row < self.rows, "Row index out of bounds");
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Iterator over a column
    pub fn col(&self, col: usize) -> Vec<T> {
        assert!(col < self.cols, "Column index out of bounds");
        (0..self.rows)
            .map(|r| self.data[r * self.cols + col].clone())
            .collect()
    }
}

#[cfg(test)]
mod tests;
