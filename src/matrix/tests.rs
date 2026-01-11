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

// Tests for arithmetic operations (addition)
mod addition_tests {
    use super::*;

    #[test]
    fn test_add_owned_owned() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a + b;
        let expected = Matrix::new(2, 2, vec![6.0, 8.0, 10.0, 12.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_add_owned_ref() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a + &b;
        let expected = Matrix::new(2, 2, vec![6.0, 8.0, 10.0, 12.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_add_ref_ref() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = &a + &b;
        let expected = Matrix::new(2, 2, vec![6.0, 8.0, 10.0, 12.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_add_different_sizes() {
        let a = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 2, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let c = a + b;
        let expected = Matrix::new(3, 2, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
        assert_eq!(c, expected);
    }

    #[test]
    #[should_panic(expected = "Row dimensions must match")]
    fn test_add_row_mismatch() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(3, 2, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let _ = a + b;
    }

    #[test]
    #[should_panic(expected = "Column dimensions must match")]
    fn test_add_col_mismatch() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 3, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let _ = a + b;
    }

    #[test]
    fn test_add_i32() {
        let a = Matrix::new(2, 2, vec![1i32, 2, 3, 4]);
        let b = Matrix::new(2, 2, vec![5i32, 6, 7, 8]);
        let c = a + b;
        let expected = Matrix::new(2, 2, vec![6i32, 8, 10, 12]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_add_f32() {
        let a = Matrix::new(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0f32, 6.0, 7.0, 8.0]);
        let c = a + b;
        let expected = Matrix::new(2, 2, vec![6.0f32, 8.0, 10.0, 12.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_add_complex() {
        use num_complex::Complex;
        let a = Matrix::new(
            2,
            2,
            vec![
                Complex::new(1.0, 2.0),
                Complex::new(3.0, 4.0),
                Complex::new(5.0, 6.0),
                Complex::new(7.0, 8.0),
            ],
        );
        let b = Matrix::new(
            2,
            2,
            vec![
                Complex::new(10.0, 1.0),
                Complex::new(20.0, 2.0),
                Complex::new(30.0, 3.0),
                Complex::new(40.0, 4.0),
            ],
        );
        let c = a + b;
        let expected = Matrix::new(
            2,
            2,
            vec![
                Complex::new(11.0, 3.0),
                Complex::new(23.0, 6.0),
                Complex::new(35.0, 9.0),
                Complex::new(47.0, 12.0),
            ],
        );
        assert_eq!(c, expected);
    }
}

// Tests for arithmetic operations (subtraction)
mod subtraction_tests {
    use super::*;

    #[test]
    fn test_sub_owned_owned() {
        let a = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let b = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a - b;
        let expected = Matrix::new(2, 2, vec![4.0, 4.0, 4.0, 4.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_sub_owned_ref() {
        let a = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let b = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a - &b;
        let expected = Matrix::new(2, 2, vec![4.0, 4.0, 4.0, 4.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_sub_ref_ref() {
        let a = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let b = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = &a - &b;
        let expected = Matrix::new(2, 2, vec![4.0, 4.0, 4.0, 4.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_sub_different_sizes() {
        let a = Matrix::new(3, 2, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let b = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = a - b;
        let expected = Matrix::new(3, 2, vec![9.0, 18.0, 27.0, 36.0, 45.0, 54.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_sub_negative_result() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a - b;
        let expected = Matrix::new(2, 2, vec![-4.0, -4.0, -4.0, -4.0]);
        assert_eq!(c, expected);
    }

    #[test]
    #[should_panic(expected = "Row dimensions must match")]
    fn test_sub_row_mismatch() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(3, 2, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let _ = a - b;
    }

    #[test]
    #[should_panic(expected = "Column dimensions must match")]
    fn test_sub_col_mismatch() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 3, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let _ = a - b;
    }

    #[test]
    fn test_sub_i32() {
        let a = Matrix::new(2, 2, vec![5i32, 6, 7, 8]);
        let b = Matrix::new(2, 2, vec![1i32, 2, 3, 4]);
        let c = a - b;
        let expected = Matrix::new(2, 2, vec![4i32, 4, 4, 4]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_sub_f32() {
        let a = Matrix::new(2, 2, vec![5.0f32, 6.0, 7.0, 8.0]);
        let b = Matrix::new(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]);
        let c = a - b;
        let expected = Matrix::new(2, 2, vec![4.0f32, 4.0, 4.0, 4.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_sub_complex() {
        use num_complex::Complex;
        let a = Matrix::new(
            2,
            2,
            vec![
                Complex::new(10.0, 5.0),
                Complex::new(20.0, 6.0),
                Complex::new(30.0, 7.0),
                Complex::new(40.0, 8.0),
            ],
        );
        let b = Matrix::new(
            2,
            2,
            vec![
                Complex::new(1.0, 2.0),
                Complex::new(3.0, 4.0),
                Complex::new(5.0, 6.0),
                Complex::new(7.0, 8.0),
            ],
        );
        let c = a - b;
        let expected = Matrix::new(
            2,
            2,
            vec![
                Complex::new(9.0, 3.0),
                Complex::new(17.0, 2.0),
                Complex::new(25.0, 1.0),
                Complex::new(33.0, 0.0),
            ],
        );
        assert_eq!(c, expected);
    }
}

// Tests for arithmetic operations (multiplication)
mod multiplication_tests {
    use super::*;

    #[test]
    fn test_mul_owned_owned() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a * b;
        // Result should be 2x2
        // [1 2 3]   [7  8 ]   [58  64]
        // [4 5 6] * [9  10] = [139 154]
        //           [11 12]
        let expected = Matrix::new(2, 2, vec![58.0, 64.0, 139.0, 154.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_mul_owned_ref() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a * &b;
        let expected = Matrix::new(2, 2, vec![58.0, 64.0, 139.0, 154.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_mul_ref_ref() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = &a * &b;
        let expected = Matrix::new(2, 2, vec![58.0, 64.0, 139.0, 154.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_mul_square_matrices() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a * b;
        // [1 2] * [5 6] = [19 22]
        // [3 4]   [7 8]   [43 50]
        let expected = Matrix::new(2, 2, vec![19.0, 22.0, 43.0, 50.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_mul_identity() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let identity = Matrix::identity(2, 2);
        let c = &a * &identity;
        assert_eq!(c, a);
    }

    #[test]
    fn test_mul_identity_left() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let identity = Matrix::identity(2, 2);
        let c = &identity * &a;
        assert_eq!(c, a);
    }

    #[test]
    fn test_mul_different_dimensions() {
        // 1x3 * 3x4 = 1x4
        let a = Matrix::new(1, 3, vec![1.0, 2.0, 3.0]);
        let b = Matrix::new(3, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a * b;
        // [1 2 3] * [1 2 3 4  ] = [38 44 50 56]
        //           [5 6 7 8  ]
        //           [9 10 11 12]
        let expected = Matrix::new(1, 4, vec![38.0, 44.0, 50.0, 56.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_mul_tall_matrices() {
        // 3x2 * 2x1 = 3x1
        let a = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(2, 1, vec![7.0, 8.0]);
        let c = a * b;
        // [1 2]   [7]   [23]
        // [3 4] * [8] = [53]
        // [5 6]         [83]
        let expected = Matrix::new(3, 1, vec![23.0, 53.0, 83.0]);
        assert_eq!(c, expected);
    }

    #[test]
    #[should_panic(expected = "Matrix dimensions incompatible for multiplication")]
    fn test_mul_incompatible_dimensions() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Matrix::new(2, 2, vec![7.0, 8.0, 9.0, 10.0]);
        let _ = a * b;
    }

    #[test]
    fn test_mul_i32() {
        let a = Matrix::new(2, 2, vec![1i32, 2, 3, 4]);
        let b = Matrix::new(2, 2, vec![5i32, 6, 7, 8]);
        let c = a * b;
        let expected = Matrix::new(2, 2, vec![19i32, 22, 43, 50]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_mul_f32() {
        let a = Matrix::new(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0f32, 6.0, 7.0, 8.0]);
        let c = a * b;
        let expected = Matrix::new(2, 2, vec![19.0f32, 22.0, 43.0, 50.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_mul_complex() {
        use num_complex::Complex;
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
                Complex::new(5.0, 0.0),
                Complex::new(6.0, 0.0),
                Complex::new(7.0, 0.0),
                Complex::new(8.0, 0.0),
            ],
        );
        let c = a * b;
        let expected = Matrix::new(
            2,
            2,
            vec![
                Complex::new(19.0, 0.0),
                Complex::new(22.0, 0.0),
                Complex::new(43.0, 0.0),
                Complex::new(50.0, 0.0),
            ],
        );
        assert_eq!(c, expected);
    }

    #[test]
    fn test_mul_complex_with_imaginary() {
        use num_complex::Complex;
        // (1+i) * (1-i) = 1 - i + i - i^2 = 1 + 1 = 2
        let a = Matrix::new(1, 1, vec![Complex::new(1.0, 1.0)]);
        let b = Matrix::new(1, 1, vec![Complex::new(1.0, -1.0)]);
        let c = a * b;
        let expected = Matrix::new(1, 1, vec![Complex::new(2.0, 0.0)]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_mul_zero_matrix() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let zero = Matrix::zero(2, 2);
        let c = a * zero;
        let expected = Matrix::zero(2, 2);
        assert_eq!(c, expected);
    }
}

// Tests for matrix exponentiation
mod exponentiation_tests {
    use super::*;

    #[test]
    fn test_pow_zero() {
        // A^0 = I (identity matrix)
        let a = Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let result = a.pow(0);
        let expected = Matrix::identity(3, 3);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_one() {
        // A^1 = A
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let result = a.pow(1);
        let expected = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_two() {
        // A^2 = A * A
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let result = a.pow(2);
        // [1 2] * [1 2] = [7  10]
        // [3 4]   [3 4]   [15 22]
        let expected = Matrix::new(2, 2, vec![7.0, 10.0, 15.0, 22.0]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_three() {
        // A^3 = A * A * A
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let result = a.pow(3);
        // A^2 = [7 10; 15 22]
        // A^3 = A^2 * A = [37 54; 81 118]
        let expected = Matrix::new(2, 2, vec![37.0, 54.0, 81.0, 118.0]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_identity() {
        // I^n = I for any n
        let identity = Matrix::<f64>::identity(3, 3);
        let result = identity.pow(5);
        let expected = Matrix::<f64>::identity(3, 3);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_zero_matrix() {
        // 0^n = 0 for n > 0
        let zero = Matrix::<f64>::zero(2, 2);
        let result = zero.pow(3);
        let expected = Matrix::<f64>::zero(2, 2);
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic(expected = "Matrix must be square for exponentiation")]
    fn test_pow_non_square_matrix() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let _ = a.pow(2);
    }

    #[test]
    fn test_pow_i32() {
        let a = Matrix::new(2, 2, vec![1i32, 2, 3, 4]);
        let result = a.pow(2);
        let expected = Matrix::new(2, 2, vec![7i32, 10, 15, 22]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_pow_complex() {
        use num_complex::Complex;
        let a = Matrix::new(
            2,
            2,
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 1.0),
                Complex::new(0.0, 1.0),
                Complex::new(1.0, 0.0),
            ],
        );
        let result = a.pow(2);
        // [1 i] * [1 i] = [1+i^2  i+i ] = [0   2i]
        // [i 1]   [i 1]   [i+i    i^2+1]  [2i  0 ]
        let expected = Matrix::new(
            2,
            2,
            vec![
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 2.0),
                Complex::new(0.0, 2.0),
                Complex::new(0.0, 0.0),
            ],
        );
        assert_eq!(result, expected);
    }
}

// Tests for dot product
mod dot_product_tests {
    use super::*;

    #[test]
    fn test_dot_vectors_row() {
        // Dot product of two row vectors
        let a = Matrix::new(1, 3, vec![1.0, 2.0, 3.0]);
        let b = Matrix::new(1, 3, vec![4.0, 5.0, 6.0]);
        let result = a.dot(&b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_dot_vectors_column() {
        // Dot product of two column vectors
        let a = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]);
        let b = Matrix::new(3, 1, vec![4.0, 5.0, 6.0]);
        let result = a.dot(&b);
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_dot_matrices() {
        // Dot product of two matrices (treated as flattened vectors)
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let result = a.dot(&b);
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert_eq!(result, 70.0);
    }

    #[test]
    fn test_dot_zero_vectors() {
        let a = Matrix::new(1, 3, vec![1.0, 2.0, 3.0]);
        let b = Matrix::new(1, 3, vec![0.0, 0.0, 0.0]);
        let result = a.dot(&b);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_dot_same_vector() {
        // Dot product of a vector with itself gives squared magnitude
        let a = Matrix::new(1, 3, vec![2.0, 3.0, 4.0]);
        let result = a.dot(&a);
        // 2^2 + 3^2 + 4^2 = 4 + 9 + 16 = 29
        assert_eq!(result, 29.0);
    }

    #[test]
    #[should_panic(expected = "Matrices must have the same total number of elements for dot product")]
    fn test_dot_different_sizes() {
        let a = Matrix::new(1, 3, vec![1.0, 2.0, 3.0]);
        let b = Matrix::new(1, 4, vec![4.0, 5.0, 6.0, 7.0]);
        let _ = a.dot(&b);
    }

    #[test]
    fn test_dot_i32() {
        let a = Matrix::new(1, 3, vec![1i32, 2, 3]);
        let b = Matrix::new(1, 3, vec![4i32, 5, 6]);
        let result = a.dot(&b);
        assert_eq!(result, 32i32);
    }

    #[test]
    fn test_dot_f32() {
        let a = Matrix::new(1, 3, vec![1.0f32, 2.0, 3.0]);
        let b = Matrix::new(1, 3, vec![4.0f32, 5.0, 6.0]);
        let result = a.dot(&b);
        assert_eq!(result, 32.0f32);
    }

    #[test]
    fn test_dot_complex() {
        use num_complex::Complex;
        let a = Matrix::new(
            1,
            2,
            vec![Complex::new(1.0, 1.0), Complex::new(2.0, 0.0)],
        );
        let b = Matrix::new(
            1,
            2,
            vec![Complex::new(1.0, -1.0), Complex::new(3.0, 0.0)],
        );
        let result = a.dot(&b);
        // (1+i)*(1-i) + 2*3 = 2 + 6 = 8
        assert_eq!(result, Complex::new(8.0, 0.0));
    }
}

// Tests for scalar multiplication
mod scalar_multiplication_tests {
    use super::*;

    #[test]
    fn test_scalar_mul_matrix_owned() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a * 2.0;
        let expected = Matrix::new(2, 2, vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_scalar_mul_matrix_ref() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = &a * 2.0;
        let expected = Matrix::new(2, 2, vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_scalar_mul_left_owned() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = 2.0 * a;
        let expected = Matrix::new(2, 2, vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_scalar_mul_left_ref() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = 2.0 * &a;
        let expected = Matrix::new(2, 2, vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_scalar_mul_commutative() {
        let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c1 = &a * 3.0;
        let c2 = 3.0 * &a;
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_scalar_mul_zero() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a * 0.0;
        let expected = Matrix::zero(2, 2);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_scalar_mul_one() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = &a * 1.0;
        assert_eq!(c, a);
    }

    #[test]
    fn test_scalar_mul_negative() {
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let c = a * -1.0;
        let expected = Matrix::new(2, 2, vec![-1.0, -2.0, -3.0, -4.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_scalar_mul_fractional() {
        let a = Matrix::new(2, 2, vec![2.0, 4.0, 6.0, 8.0]);
        let c = a * 0.5;
        let expected = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_scalar_mul_i32() {
        let a = Matrix::new(2, 2, vec![1i32, 2, 3, 4]);
        let c = a * 3i32;
        let expected = Matrix::new(2, 2, vec![3i32, 6, 9, 12]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_scalar_mul_i32_left() {
        let a = Matrix::new(2, 2, vec![1i32, 2, 3, 4]);
        let c = 3i32 * a;
        let expected = Matrix::new(2, 2, vec![3i32, 6, 9, 12]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_scalar_mul_f32() {
        let a = Matrix::new(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]);
        let c = a * 2.5f32;
        let expected = Matrix::new(2, 2, vec![2.5f32, 5.0, 7.5, 10.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_scalar_mul_f32_left() {
        let a = Matrix::new(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]);
        let c = 2.5f32 * a;
        let expected = Matrix::new(2, 2, vec![2.5f32, 5.0, 7.5, 10.0]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_scalar_mul_different_sizes() {
        let a = Matrix::new(3, 4, vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0
        ]);
        let c = &a * 2.0;
        let expected = Matrix::new(3, 4, vec![
            2.0, 4.0, 6.0, 8.0,
            10.0, 12.0, 14.0, 16.0,
            18.0, 20.0, 22.0, 24.0
        ]);
        assert_eq!(c, expected);
    }

    #[test]
    fn test_scalar_mul_complex() {
        use num_complex::Complex;
        let a = Matrix::new(
            2,
            2,
            vec![
                Complex::new(1.0, 2.0),
                Complex::new(3.0, 4.0),
                Complex::new(5.0, 6.0),
                Complex::new(7.0, 8.0),
            ],
        );
        let c = a * Complex::new(2.0, 0.0);
        let expected = Matrix::new(
            2,
            2,
            vec![
                Complex::new(2.0, 4.0),
                Complex::new(6.0, 8.0),
                Complex::new(10.0, 12.0),
                Complex::new(14.0, 16.0),
            ],
        );
        assert_eq!(c, expected);
    }

    #[test]
    fn test_scalar_mul_complex_imaginary() {
        use num_complex::Complex;
        // Multiply by i: (a+bi) * i = -b + ai
        let a = Matrix::new(
            2,
            2,
            vec![
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 1.0),
                Complex::new(2.0, 3.0),
                Complex::new(4.0, -1.0),
            ],
        );
        let c = a * Complex::new(0.0, 1.0);
        let expected = Matrix::new(
            2,
            2,
            vec![
                Complex::new(0.0, 1.0),
                Complex::new(-1.0, 0.0),
                Complex::new(-3.0, 2.0),
                Complex::new(1.0, 4.0),
            ],
        );
        assert_eq!(c, expected);
    }
}

// Tests for outer product
mod outer_product_tests {
    use super::*;

    #[test]
    fn test_outer_basic() {
        // Outer product of two vectors
        let a = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]);
        let b = Matrix::new(2, 1, vec![4.0, 5.0]);
        let result = a.outer(&b);
        // [1]     [4 5]
        // [2] ⊗ = [8 10]
        // [3]     [12 15]
        let expected = Matrix::new(3, 2, vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_outer_row_vectors() {
        let a = Matrix::new(1, 3, vec![1.0, 2.0, 3.0]);
        let b = Matrix::new(1, 2, vec![4.0, 5.0]);
        let result = a.outer(&b);
        let expected = Matrix::new(3, 2, vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_outer_different_sizes() {
        let a = Matrix::new(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(1, 2, vec![5.0, 6.0]);
        let result = a.outer(&b);
        // [1 2 3 4] ⊗ [5 6] =
        // [5  6 ]
        // [10 12]
        // [15 18]
        // [20 24]
        let expected = Matrix::new(4, 2, vec![5.0, 6.0, 10.0, 12.0, 15.0, 18.0, 20.0, 24.0]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_outer_with_zero() {
        let a = Matrix::new(1, 2, vec![1.0, 2.0]);
        let b = Matrix::new(1, 2, vec![0.0, 0.0]);
        let result = a.outer(&b);
        let expected = Matrix::zero(2, 2);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_outer_single_elements() {
        let a = Matrix::new(1, 1, vec![3.0]);
        let b = Matrix::new(1, 1, vec![4.0]);
        let result = a.outer(&b);
        let expected = Matrix::new(1, 1, vec![12.0]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_outer_matrices_as_vectors() {
        // Outer product treating matrices as flattened vectors
        let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::new(1, 2, vec![5.0, 6.0]);
        let result = a.outer(&b);
        // [1 2 3 4] ⊗ [5 6] produces a 4x2 matrix
        let expected = Matrix::new(4, 2, vec![5.0, 6.0, 10.0, 12.0, 15.0, 18.0, 20.0, 24.0]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_outer_i32() {
        let a = Matrix::new(1, 3, vec![1i32, 2, 3]);
        let b = Matrix::new(1, 2, vec![4i32, 5]);
        let result = a.outer(&b);
        let expected = Matrix::new(3, 2, vec![4i32, 5, 8, 10, 12, 15]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_outer_f32() {
        let a = Matrix::new(1, 2, vec![2.0f32, 3.0]);
        let b = Matrix::new(1, 2, vec![4.0f32, 5.0]);
        let result = a.outer(&b);
        let expected = Matrix::new(2, 2, vec![8.0f32, 10.0, 12.0, 15.0]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_outer_complex() {
        use num_complex::Complex;
        let a = Matrix::new(
            1,
            2,
            vec![Complex::new(1.0, 1.0), Complex::new(2.0, 0.0)],
        );
        let b = Matrix::new(
            1,
            2,
            vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)],
        );
        let result = a.outer(&b);
        // [1+i  i-1]
        // [2    2i ]
        let expected = Matrix::new(
            2,
            2,
            vec![
                Complex::new(1.0, 1.0),
                Complex::new(-1.0, 1.0),
                Complex::new(2.0, 0.0),
                Complex::new(0.0, 2.0),
            ],
        );
        assert_eq!(result, expected);
    }
}
