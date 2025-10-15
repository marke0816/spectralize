#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Create a new matrix from a Vec<f64>
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len());
        Self { rows, cols, data }
    }

    /// Create a rows x cols zero matrix
    pub fn zero(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Create a rows x cols identity matrix
    pub fn identity(rows: usize, cols: usize) -> Self {
        let mut data = vec![0.0; rows * cols];

        (0..rows.min(cols)).for_each(|i| data[i * cols + i] = 1.0);

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
        let mut data = vec![0.0; rows * cols];

        for (row, &col_index) in perm.iter().enumerate() {
            assert!(
                col_index >= 1 && col_index <= cols,
                "Column indices must be 1-based and <= cols"
            );
            data[row * cols + (col_index - 1)] = 1.0;
        }

        Self { rows, cols, data }
    }

    /// Returns a new matrix formed by concatenating `self` and `other` horizontally.
    pub fn with_cols(&self, other: &Matrix) -> Self {
        assert_eq!(self.rows, other.rows, "Row counts must match");

        let mut new_data = Vec::with_capacity(self.rows * (self.cols + other.cols));

        for r in 0..self.rows {
            let start_a = r * self.cols;
            new_data.extend_from_slice(&self.data[start_a..start_a + self.cols]);

            let start_b = r * other.cols;
            new_data.extend_from_slice(&other.data[start_b..start_b + other.cols]);
        }

        Self {
            rows: self.rows,
            cols: self.cols + other.cols,
            data: new_data,
        }
    }

    /// Returns a new matrix formed by concatenating `self` and `other` vertically.
    pub fn with_rows(&self, other: &Matrix) -> Self {
        assert_eq!(self.cols, other.cols, "Column counts must match");

        let mut new_data = Vec::with_capacity((self.rows + other.rows) * self.cols);
        new_data.extend_from_slice(&self.data);
        new_data.extend_from_slice(&other.data);

        Self {
            rows: self.rows + other.rows,
            cols: self.cols,
            data: new_data,
        }
    }

    /// Returns a new matrix with a row (given as a slice) appended at the bottom.
    pub fn with_row_vec(&self, row: &[f64]) -> Self {
        assert_eq!(self.cols, row.len(), "Row length must match column count");

        let mut new_data = Vec::with_capacity((self.rows + 1) * self.cols);
        new_data.extend_from_slice(&self.data);
        new_data.extend_from_slice(row);

        Self {
            rows: self.rows + 1,
            cols: self.cols,
            data: new_data,
        }
    }

    /// Returns a new matrix with a column (given as a slice) appended to the right.
    pub fn with_col_vec(&self, col: &[f64]) -> Self {
        assert_eq!(self.rows, col.len(), "Column length must match row count");

        let mut new_data = Vec::with_capacity(self.rows * (self.cols + 1));

        for r in 0..self.rows {
            let start = r * self.cols;
            new_data.extend_from_slice(&self.data[start..start + self.cols]);
            new_data.push(col[r]);
        }

        Self {
            rows: self.rows,
            cols: self.cols + 1,
            data: new_data,
        }
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
    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        self.data[row * self.cols + col]
    }

    /// Safe setter
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        self.data[row * self.cols + col] = value;
    }

    /// Iterator over a row
    pub fn row(&self, row: usize) -> &[f64] {
        assert!(row < self.rows, "Row index out of bounds");
        let start = row * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Iterator over a column
    pub fn col(&self, col: usize) -> Vec<f64> {
        assert!(col < self.cols, "Column index out of bounds");
        (0..self.rows)
            .map(|r| self.data[r * self.cols + col])
            .collect()
    }
}

#[cfg(test)]
mod tests;
