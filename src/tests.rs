use super::*;

fn sample_matrix() -> Matrix {
    Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
}

#[test]
fn test_matrix_creation() {
    let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(m.rows, 2);
    assert_eq!(m.cols, 3);
    assert_eq!(m.data.len(), 6);
    assert_eq!(m.get(0, 0), 1.0);
    assert_eq!(m.get(1, 2), 6.0);
}

#[test]
#[should_panic(expected = "left == right")]
fn test_matrix_creation_with_wrong_data_length_panics() {
    // 2x3 matrix but only 5 data elements — should panic
    Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_matrix_get() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(m.get(0, 0), 1.0);
    assert_eq!(m.get(0, 1), 2.0);
    assert_eq!(m.get(1, 0), 3.0);
    assert_eq!(m.get(1, 1), 4.0);
}

#[test]
fn test_matrix_set() {
    let mut m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    m.set(0, 1, 9.0);
    m.set(1, 0, 8.0);
    assert_eq!(m.get(0, 1), 9.0);
    assert_eq!(m.get(1, 0), 8.0);
}

#[test]
fn test_matrix_row_major_order() {
    // 2x3 matrix stored row-major: [r0c0, r0c1, r0c2, r1c0, r1c1, r1c2]
    let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(m.get(0, 0), 1.0);
    assert_eq!(m.get(0, 2), 3.0);
    assert_eq!(m.get(1, 0), 4.0);
    assert_eq!(m.get(1, 2), 6.0);
}

#[test]
fn test_matrix_basic() {
    let mut m = Matrix::zero(2, 3);
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);

    m.set(0, 1, 5.0);
    assert_eq!(m.get(0, 1), 5.0);

    let row0 = m.row(0);
    assert_eq!(row0, &[0.0, 5.0, 0.0]);

    let col1 = m.col(1);
    assert_eq!(col1, vec![5.0, 0.0]);
}

#[test]
#[should_panic(expected = "Index out of bounds")]
fn test_out_of_bounds_get() {
    let m = Matrix::zero(2, 2);
    m.get(0, 3);
}

#[test]
fn test_identity_square() {
    let m = Matrix::identity(3, 3);
    let expected = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    assert_eq!(m.rows, 3);
    assert_eq!(m.cols, 3);
    assert_eq!(m.data, expected);
}

#[test]
fn test_identity_rectangular_rows_greater() {
    let m = Matrix::identity(4, 3);
    let expected = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    assert_eq!(m.rows, 4);
    assert_eq!(m.cols, 3);
    assert_eq!(m.data, expected);
}

#[test]
fn test_identity_rectangular_cols_greater() {
    let m = Matrix::identity(3, 5);
    let expected = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    ];
    assert_eq!(m.rows, 3);
    assert_eq!(m.cols, 5);
    assert_eq!(m.data, expected);
}

#[test]
fn test_identity_zero_rows_or_cols() {
    let m = Matrix::identity(0, 3);
    assert_eq!(m.rows, 0);
    assert_eq!(m.cols, 3);
    assert!(m.data.is_empty());

    let m = Matrix::identity(3, 0);
    assert_eq!(m.rows, 3);
    assert_eq!(m.cols, 0);
    assert!(m.data.is_empty());
}

#[test]
fn test_perm_square() {
    let m = Matrix::perm(4, 4, vec![2, 4, 3, 1]);
    let expected = vec![
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(m.rows, 4);
    assert_eq!(m.cols, 4);
    assert_eq!(m.data, expected);
}

#[test]
fn test_perm_rectangular() {
    let m = Matrix::perm(3, 5, vec![1, 5, 3]);
    let expected = vec![
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    ];
    assert_eq!(m.rows, 3);
    assert_eq!(m.cols, 5);
    assert_eq!(m.data, expected);
}

#[test]
#[should_panic(expected = "Length of permutation vector must match rows")]
fn test_perm_invalid_length() {
    // perm vector length != rows
    Matrix::perm(3, 3, vec![1, 2]);
}

#[test]
#[should_panic(expected = "Column indices must be 1-based and <= cols")]
fn test_perm_invalid_index_zero() {
    // column index 0 is invalid (1-based)
    Matrix::perm(3, 3, vec![1, 0, 2]);
}

#[test]
#[should_panic(expected = "Column indices must be 1-based and <= cols")]
fn test_perm_invalid_index_out_of_bounds() {
    // column index greater than cols is invalid
    Matrix::perm(3, 3, vec![1, 2, 4]);
}

#[test]
fn test_get_valid_indices() {
    let m = sample_matrix();
    assert_eq!(m.get(0, 0), 1.0);
    assert_eq!(m.get(1, 2), 6.0);
    assert_eq!(m.get(2, 1), 8.0);
}

#[test]
#[should_panic(expected = "Index out of bounds")]
fn test_get_out_of_bounds_row() {
    let m = sample_matrix();
    let _ = m.get(3, 0);
}

#[test]
#[should_panic(expected = "Index out of bounds")]
fn test_get_out_of_bounds_col() {
    let m = sample_matrix();
    let _ = m.get(0, 3);
}

#[test]
fn test_set_valid_indices() {
    let mut m = sample_matrix();
    m.set(1, 1, 42.0);
    assert_eq!(m.get(1, 1), 42.0);
}

#[test]
#[should_panic(expected = "Index out of bounds")]
fn test_set_out_of_bounds() {
    let mut m = sample_matrix();
    m.set(5, 0, 99.0);
}

#[test]
fn test_row_valid() {
    let m = sample_matrix();
    let row = m.row(1);
    assert_eq!(row, &[4.0, 5.0, 6.0]);
}

#[test]
#[should_panic(expected = "Row index out of bounds")]
fn test_row_out_of_bounds() {
    let m = sample_matrix();
    let _ = m.row(3);
}

#[test]
fn test_col_valid() {
    let m = sample_matrix();
    let col = m.col(2);
    assert_eq!(col, vec![3.0, 6.0, 9.0]);
}

#[test]
#[should_panic(expected = "Column index out of bounds")]
fn test_col_out_of_bounds() {
    let m = sample_matrix();
    let _ = m.col(5);
}

#[test]
fn test_with_cols() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

    let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

    let c = a.with_cols(&b);

    assert_eq!(c.rows, 2);
    assert_eq!(c.cols, 4);
    assert_eq!(c.data, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0,]);
}

#[test]
#[should_panic(expected = "Row counts must match")]
fn test_with_cols_mismatch() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let b = Matrix::new(3, 2, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    a.with_cols(&b);
}

#[test]
fn test_with_rows() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

    let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

    let c = a.with_rows(&b);

    assert_eq!(c.rows, 4);
    assert_eq!(c.cols, 2);
    assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,]);
}

#[test]
#[should_panic(expected = "Column counts must match")]
fn test_with_rows_mismatch() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let b = Matrix::new(2, 3, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    a.with_rows(&b);
}

#[test]
fn test_with_row_vec() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

    let c = a.with_row_vec(&[5.0, 6.0]);

    assert_eq!(c.rows, 3);
    assert_eq!(c.cols, 2);
    assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0,]);
}

#[test]
#[should_panic(expected = "Row length must match column count")]
fn test_with_row_vec_mismatch() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    a.with_row_vec(&[5.0, 6.0, 7.0]); // wrong length
}

#[test]
fn test_with_col_vec() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

    let c = a.with_col_vec(&[5.0, 6.0]);

    assert_eq!(c.rows, 2);
    assert_eq!(c.cols, 3);
    assert_eq!(c.data, vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0,]);
}

#[test]
#[should_panic(expected = "Column length must match row count")]
fn test_with_col_vec_mismatch() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    a.with_col_vec(&[5.0, 6.0, 7.0]); // wrong length
}
