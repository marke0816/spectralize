use super::*;

fn sample_matrix() -> Matrix {
    Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
}

#[test]
fn test_matrix_creation() {
    let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let expected = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(m, expected);
}

#[test]
#[should_panic]
fn test_matrix_creation_with_wrong_data_length_panics() {
    Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_matrix_get_set() {
    let mut m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    m.set(0, 1, 9.0);
    m.set(1, 0, 8.0);
    let expected = Matrix::new(2, 2, vec![1.0, 9.0, 8.0, 4.0]);
    assert_eq!(m, expected);
}

#[test]
fn test_row_and_col() {
    let m = sample_matrix();
    assert_eq!(m.row(1), &[4.0, 5.0, 6.0]);
    assert_eq!(m.col(2), vec![3.0, 6.0, 9.0]);
}

#[test]
#[should_panic]
fn test_row_out_of_bounds() {
    sample_matrix().row(3);
}

#[test]
#[should_panic]
fn test_col_out_of_bounds() {
    sample_matrix().col(5);
}

#[test]
fn test_identity_matrices() {
    let m = Matrix::identity(3, 3);
    let expected = Matrix::new(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    assert_eq!(m, expected);

    let m = Matrix::identity(4, 3);
    let expected = Matrix::new(
        4,
        3,
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    );
    assert_eq!(m, expected);

    let m = Matrix::identity(3, 5);
    let expected = Matrix::new(
        3,
        5,
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ],
    );
    assert_eq!(m, expected);
}

#[test]
fn test_perm_matrices() {
    let m = Matrix::perm(4, 4, vec![2, 4, 3, 1]);
    let expected = Matrix::new(
        4,
        4,
        vec![
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        ],
    );
    assert_eq!(m, expected);

    let m = Matrix::perm(3, 5, vec![1, 5, 3]);
    let expected = Matrix::new(
        3,
        5,
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ],
    );
    assert_eq!(m, expected);
}

#[test]
#[should_panic]
fn test_perm_invalid_length() {
    Matrix::perm(3, 3, vec![1, 2]);
}

#[test]
#[should_panic]
fn test_perm_invalid_index_zero() {
    Matrix::perm(3, 3, vec![1, 0, 2]);
}

#[test]
#[should_panic]
fn test_perm_invalid_index_out_of_bounds() {
    Matrix::perm(3, 3, vec![1, 2, 4]);
}

#[test]
fn test_with_cols_and_rows() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
    let c = a.with_cols(&b);
    let expected = Matrix::new(2, 4, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    assert_eq!(c, expected);

    let c = a.with_rows(&b);
    let expected = Matrix::new(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(c, expected);
}

#[test]
#[should_panic]
fn test_with_cols_mismatch() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let b = Matrix::new(3, 2, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    a.with_cols(&b);
}

#[test]
#[should_panic]
fn test_with_rows_mismatch() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let b = Matrix::new(2, 3, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    a.with_rows(&b);
}

#[test]
fn test_with_row_vec_and_col_vec() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let c = a.with_row_vec(&[5.0, 6.0]);
    let expected = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(c, expected);

    let c = a.with_col_vec(&[5.0, 6.0]);
    let expected = Matrix::new(2, 3, vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    assert_eq!(c, expected);
}

#[test]
#[should_panic]
fn test_with_row_vec_mismatch() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    a.with_row_vec(&[5.0, 6.0, 7.0]);
}

#[test]
#[should_panic]
fn test_with_col_vec_mismatch() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    a.with_col_vec(&[5.0, 6.0, 7.0]);
}

