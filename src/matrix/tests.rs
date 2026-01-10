use super::*;

fn sample_matrix() -> Matrix<f64> {
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
    Matrix::<f64>::perm(3, 3, vec![1, 2]);
}

#[test]
#[should_panic]
fn test_perm_invalid_index_zero() {
    Matrix::<f64>::perm(3, 3, vec![1, 0, 2]);
}

#[test]
#[should_panic]
fn test_perm_invalid_index_out_of_bounds() {
    Matrix::<f64>::perm(3, 3, vec![1, 2, 4]);
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

// Tests for f32 matrices
mod f32_tests {
    use super::*;

    #[test]
    fn test_f32_matrix_creation() {
        let m = Matrix::new(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(m.get(0, 0), 1.0f32);
        assert_eq!(m.get(1, 1), 4.0f32);
    }

    #[test]
    fn test_f32_identity() {
        let m = Matrix::<f32>::identity(3, 3);
        assert_eq!(m.get(0, 0), 1.0f32);
        assert_eq!(m.get(0, 1), 0.0f32);
        assert_eq!(m.get(1, 1), 1.0f32);
    }

    #[test]
    fn test_f32_zero() {
        let m = Matrix::<f32>::zero(2, 3);
        assert_eq!(m.get(0, 0), 0.0f32);
        assert_eq!(m.get(1, 2), 0.0f32);
    }

    #[test]
    fn test_f32_concatenation() {
        let a = Matrix::new(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0f32, 6.0, 7.0, 8.0]);
        let c = a.with_cols(&b);
        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 4);
        assert_eq!(c.get(0, 2), 5.0f32);
    }
}

// Tests for i32 matrices
mod i32_tests {
    use super::*;

    #[test]
    fn test_i32_matrix_creation() {
        let m = Matrix::new(2, 2, vec![1i32, 2, 3, 4]);
        assert_eq!(m.get(0, 0), 1i32);
        assert_eq!(m.get(1, 1), 4i32);
    }

    #[test]
    fn test_i32_identity() {
        let m = Matrix::<i32>::identity(3, 3);
        assert_eq!(m.get(0, 0), 1i32);
        assert_eq!(m.get(0, 1), 0i32);
        assert_eq!(m.get(1, 1), 1i32);
    }

    #[test]
    fn test_i32_zero() {
        let m = Matrix::<i32>::zero(2, 3);
        assert_eq!(m.get(0, 0), 0i32);
        assert_eq!(m.get(1, 2), 0i32);
    }

    #[test]
    fn test_i32_perm() {
        let m = Matrix::<i32>::perm(3, 3, vec![2, 3, 1]);
        assert_eq!(m.get(0, 0), 0i32);
        assert_eq!(m.get(0, 1), 1i32);
        assert_eq!(m.get(2, 0), 1i32);
    }

    #[test]
    fn test_i32_concatenation() {
        let a = Matrix::new(2, 2, vec![1i32, 2, 3, 4]);
        let b = Matrix::new(2, 2, vec![5i32, 6, 7, 8]);
        let c = a.with_rows(&b);
        assert_eq!(c.rows(), 4);
        assert_eq!(c.cols(), 2);
        assert_eq!(c.get(2, 0), 5i32);
    }
}

// Tests for complex matrices
mod complex_tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_complex_matrix_creation() {
        let m = Matrix::new(
            2,
            2,
            vec![
                Complex::new(1.0, 2.0),
                Complex::new(3.0, 4.0),
                Complex::new(5.0, 6.0),
                Complex::new(7.0, 8.0),
            ],
        );
        assert_eq!(m.get(0, 0), Complex::new(1.0, 2.0));
        assert_eq!(m.get(1, 1), Complex::new(7.0, 8.0));
    }

    #[test]
    fn test_complex_identity() {
        let m = Matrix::<Complex<f64>>::identity(3, 3);
        assert_eq!(m.get(0, 0), Complex::new(1.0, 0.0));
        assert_eq!(m.get(0, 1), Complex::new(0.0, 0.0));
        assert_eq!(m.get(1, 1), Complex::new(1.0, 0.0));
        assert_eq!(m.get(2, 2), Complex::new(1.0, 0.0));
    }

    #[test]
    fn test_complex_zero() {
        let m = Matrix::<Complex<f64>>::zero(2, 2);
        assert_eq!(m.get(0, 0), Complex::new(0.0, 0.0));
        assert_eq!(m.get(1, 1), Complex::new(0.0, 0.0));
    }

    #[test]
    fn test_complex_perm() {
        let m = Matrix::<Complex<f64>>::perm(3, 3, vec![2, 3, 1]);
        assert_eq!(m.get(0, 0), Complex::new(0.0, 0.0));
        assert_eq!(m.get(0, 1), Complex::new(1.0, 0.0));
        assert_eq!(m.get(1, 2), Complex::new(1.0, 0.0));
    }

    #[test]
    fn test_complex_concatenation() {
        let a = Matrix::new(
            2,
            2,
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
            ],
        );
        let b = Matrix::new(
            2,
            2,
            vec![
                Complex::new(5.0, 1.0),
                Complex::new(6.0, 1.0),
                Complex::new(7.0, 1.0),
                Complex::new(8.0, 1.0),
            ],
        );
        let c = a.with_cols(&b);
        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 4);
        assert_eq!(c.get(0, 2), Complex::new(5.0, 1.0));
    }

    #[test]
    fn test_complex_row_and_col() {
        let m = Matrix::new(
            2,
            3,
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
                Complex::new(5.0, 0.0),
                Complex::new(6.0, 0.0),
            ],
        );
        let row = m.row(0);
        assert_eq!(row[0], Complex::new(1.0, 0.0));
        assert_eq!(row[2], Complex::new(3.0, 0.0));

        let col = m.col(1);
        assert_eq!(col[0], Complex::new(2.0, 0.0));
        assert_eq!(col[1], Complex::new(5.0, 0.0));
    }

    #[test]
    fn test_complex_with_vectors() {
        let a = Matrix::new(
            2,
            2,
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(2.0, 0.0),
                Complex::new(3.0, 0.0),
                Complex::new(4.0, 0.0),
            ],
        );

        let c = a.with_row_vec(&[Complex::new(5.0, 1.0), Complex::new(6.0, 1.0)]);
        assert_eq!(c.rows(), 3);
        assert_eq!(c.get(2, 0), Complex::new(5.0, 1.0));

        let c = a.with_col_vec(&[Complex::new(5.0, 1.0), Complex::new(6.0, 1.0)]);
        assert_eq!(c.cols(), 3);
        assert_eq!(c.get(0, 2), Complex::new(5.0, 1.0));
    }
}
