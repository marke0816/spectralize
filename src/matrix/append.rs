use super::Matrix;

impl Matrix {
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
}
