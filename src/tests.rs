use super::*;

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
