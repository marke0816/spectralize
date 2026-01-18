//! Tests for ConstMatrix

use super::const_matrix::ConstMatrix;
use super::{Matrix, MatrixError};

#[test]
fn new_creates_matrix_with_correct_data() {
    let m: ConstMatrix<i32, 2, 3> = ConstMatrix::new(vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(m.get(0, 0), 1);
    assert_eq!(m.get(0, 1), 2);
    assert_eq!(m.get(0, 2), 3);
    assert_eq!(m.get(1, 0), 4);
    assert_eq!(m.get(1, 1), 5);
    assert_eq!(m.get(1, 2), 6);
}

#[test]
#[should_panic(expected = "Data length")]
fn new_panics_on_wrong_size() {
    let _m: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3]); // Too short
}

#[test]
fn zero_creates_all_zeros() {
    let m: ConstMatrix<f64, 3, 3> = ConstMatrix::zero();
    for row in 0..3 {
        for col in 0..3 {
            assert_eq!(m.get(row, col), 0.0);
        }
    }
}

#[test]
fn identity_creates_diagonal_ones() {
    let m: ConstMatrix<i32, 3, 3> = ConstMatrix::identity();
    assert_eq!(m.get(0, 0), 1);
    assert_eq!(m.get(1, 1), 1);
    assert_eq!(m.get(2, 2), 1);
    assert_eq!(m.get(0, 1), 0);
    assert_eq!(m.get(1, 0), 0);
}

#[test]
fn identity_rectangular() {
    let m: ConstMatrix<i32, 2, 3> = ConstMatrix::identity();
    assert_eq!(m.get(0, 0), 1);
    assert_eq!(m.get(1, 1), 1);
    assert_eq!(m.get(0, 1), 0);
    assert_eq!(m.get(0, 2), 0);
    assert_eq!(m.get(1, 2), 0);
}

#[test]
fn rows_cols_return_const_values() {
    let m: ConstMatrix<i32, 5, 7> = ConstMatrix::zero();
    assert_eq!(m.rows(), 5);
    assert_eq!(m.cols(), 7);
}

#[test]
fn get_ref_returns_reference() {
    let m: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    assert_eq!(*m.get_ref(0, 0), 1);
    assert_eq!(*m.get_ref(1, 1), 4);
}

#[test]
fn set_modifies_element() {
    let mut m: ConstMatrix<i32, 2, 2> = ConstMatrix::zero();
    m.set(0, 1, 42);
    assert_eq!(m.get(0, 1), 42);
    assert_eq!(m.get(0, 0), 0); // Other elements unchanged
}

#[test]
fn row_returns_slice() {
    let m: ConstMatrix<i32, 2, 3> = ConstMatrix::new(vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(m.row(0), &[1, 2, 3]);
    assert_eq!(m.row(1), &[4, 5, 6]);
}

#[test]
fn col_iter_yields_column_elements() {
    let m: ConstMatrix<i32, 3, 2> = ConstMatrix::new(vec![1, 2, 3, 4, 5, 6]);
    let col: Vec<i32> = m.col_iter(0).cloned().collect();
    assert_eq!(col, vec![1, 3, 5]);

    let col: Vec<i32> = m.col_iter(1).cloned().collect();
    assert_eq!(col, vec![2, 4, 6]);
}

#[test]
fn transpose_swaps_dimensions() {
    let m: ConstMatrix<i32, 2, 3> = ConstMatrix::new(vec![1, 2, 3, 4, 5, 6]);
    let mt: ConstMatrix<i32, 3, 2> = m.transpose();

    assert_eq!(mt.rows(), 3);
    assert_eq!(mt.cols(), 2);
    assert_eq!(mt.get(0, 0), 1);
    assert_eq!(mt.get(0, 1), 4);
    assert_eq!(mt.get(1, 0), 2);
    assert_eq!(mt.get(1, 1), 5);
    assert_eq!(mt.get(2, 0), 3);
    assert_eq!(mt.get(2, 1), 6);
}

#[test]
fn transpose_square_matrix() {
    let m: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let mt = m.transpose();
    assert_eq!(mt.get(0, 0), 1);
    assert_eq!(mt.get(0, 1), 3);
    assert_eq!(mt.get(1, 0), 2);
    assert_eq!(mt.get(1, 1), 4);
}

#[test]
fn data_returns_underlying_vec() {
    let m: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    assert_eq!(m.data(), &[1, 2, 3, 4]);
}

#[test]
fn into_data_consumes_and_returns_vec() {
    let m: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let data = m.into_data();
    assert_eq!(data, vec![1, 2, 3, 4]);
}

#[test]
fn from_const_to_dynamic_matrix() {
    let cm: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let dm: Matrix<i32> = cm.into();
    assert_eq!(dm.rows(), 2);
    assert_eq!(dm.cols(), 2);
    assert_eq!(dm.get(0, 0), 1);
    assert_eq!(dm.get(1, 1), 4);
}

#[test]
fn try_from_dynamic_to_const_matrix_success() {
    let dm = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    let cm: ConstMatrix<i32, 2, 2> = dm.try_into().unwrap();
    assert_eq!(cm.get(0, 0), 1);
    assert_eq!(cm.get(1, 1), 4);
}

#[test]
fn try_from_dynamic_to_const_matrix_dimension_mismatch() {
    let dm = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    let result: Result<ConstMatrix<i32, 3, 3>, _> = dm.try_into();
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), MatrixError::DimensionMismatch);
}

#[test]
fn works_with_f64() {
    let m: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.5, 2.5, 3.5, 4.5]);
    assert_eq!(m.get(0, 0), 1.5);
    assert_eq!(m.get(1, 1), 4.5);
}

#[test]
fn works_with_complex() {
    use num_complex::Complex;
    let m: ConstMatrix<Complex<f64>, 2, 2> = ConstMatrix::new(vec![
        Complex::new(1.0, 2.0),
        Complex::new(3.0, 4.0),
        Complex::new(5.0, 6.0),
        Complex::new(7.0, 8.0),
    ]);
    assert_eq!(m.get(0, 0), Complex::new(1.0, 2.0));
    assert_eq!(m.get(1, 1), Complex::new(7.0, 8.0));
}

#[test]
fn zero_sized_matrices() {
    let m: ConstMatrix<i32, 0, 5> = ConstMatrix::zero();
    assert_eq!(m.rows(), 0);
    assert_eq!(m.cols(), 5);
    assert_eq!(m.data().len(), 0);

    let m: ConstMatrix<i32, 3, 0> = ConstMatrix::zero();
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 0);
    assert_eq!(m.data().len(), 0);
}

// ============================================================================
// Arithmetic Operation Tests
// ============================================================================

// Addition tests
#[test]
fn add_owned_owned() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let b: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![5, 6, 7, 8]);
    let c = a + b;
    assert_eq!(c.get(0, 0), 6);
    assert_eq!(c.get(0, 1), 8);
    assert_eq!(c.get(1, 0), 10);
    assert_eq!(c.get(1, 1), 12);
}

#[test]
fn add_owned_ref() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let b: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![5, 6, 7, 8]);
    let c = a + &b;
    assert_eq!(c.get(0, 0), 6);
    assert_eq!(c.get(1, 1), 12);
    // b should still be usable
    assert_eq!(b.get(0, 0), 5);
}

#[test]
fn add_ref_ref() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let b: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![5, 6, 7, 8]);
    let c = &a + &b;
    assert_eq!(c.get(0, 0), 6);
    assert_eq!(c.get(1, 1), 12);
    // Both should still be usable
    assert_eq!(a.get(0, 0), 1);
    assert_eq!(b.get(0, 0), 5);
}

#[test]
fn add_f64() {
    let a: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.5, 2.5, 3.5, 4.5]);
    let b: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![0.5, 0.5, 0.5, 0.5]);
    let c = a + b;
    assert_eq!(c.get(0, 0), 2.0);
    assert_eq!(c.get(1, 1), 5.0);
}

#[test]
fn add_complex() {
    use num_complex::Complex;
    let a: ConstMatrix<Complex<f64>, 2, 2> = ConstMatrix::new(vec![
        Complex::new(1.0, 2.0),
        Complex::new(3.0, 4.0),
        Complex::new(5.0, 6.0),
        Complex::new(7.0, 8.0),
    ]);
    let b: ConstMatrix<Complex<f64>, 2, 2> = ConstMatrix::new(vec![
        Complex::new(1.0, 1.0),
        Complex::new(2.0, 2.0),
        Complex::new(3.0, 3.0),
        Complex::new(4.0, 4.0),
    ]);
    let c = a + b;
    assert_eq!(c.get(0, 0), Complex::new(2.0, 3.0));
    assert_eq!(c.get(1, 1), Complex::new(11.0, 12.0));
}

// Subtraction tests
#[test]
fn sub_owned_owned() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![10, 20, 30, 40]);
    let b: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let c = a - b;
    assert_eq!(c.get(0, 0), 9);
    assert_eq!(c.get(0, 1), 18);
    assert_eq!(c.get(1, 0), 27);
    assert_eq!(c.get(1, 1), 36);
}

#[test]
fn sub_owned_ref() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![10, 20, 30, 40]);
    let b: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let c = a - &b;
    assert_eq!(c.get(0, 0), 9);
    assert_eq!(c.get(1, 1), 36);
    // b should still be usable
    assert_eq!(b.get(0, 0), 1);
}

#[test]
fn sub_ref_ref() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![10, 20, 30, 40]);
    let b: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let c = &a - &b;
    assert_eq!(c.get(0, 0), 9);
    assert_eq!(c.get(1, 1), 36);
    // Both should still be usable
    assert_eq!(a.get(0, 0), 10);
    assert_eq!(b.get(0, 0), 1);
}

#[test]
fn sub_f64() {
    let a: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![5.5, 6.5, 7.5, 8.5]);
    let b: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![0.5, 0.5, 0.5, 0.5]);
    let c = a - b;
    assert_eq!(c.get(0, 0), 5.0);
    assert_eq!(c.get(1, 1), 8.0);
}

#[test]
fn sub_complex() {
    use num_complex::Complex;
    let a: ConstMatrix<Complex<f64>, 2, 2> = ConstMatrix::new(vec![
        Complex::new(5.0, 6.0),
        Complex::new(7.0, 8.0),
        Complex::new(9.0, 10.0),
        Complex::new(11.0, 12.0),
    ]);
    let b: ConstMatrix<Complex<f64>, 2, 2> = ConstMatrix::new(vec![
        Complex::new(1.0, 2.0),
        Complex::new(3.0, 4.0),
        Complex::new(5.0, 6.0),
        Complex::new(7.0, 8.0),
    ]);
    let c = a - b;
    assert_eq!(c.get(0, 0), Complex::new(4.0, 4.0));
    assert_eq!(c.get(1, 1), Complex::new(4.0, 4.0));
}

// Matrix multiplication tests
#[test]
fn mul_square_matrices() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let b: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![5, 6, 7, 8]);
    let c: ConstMatrix<i32, 2, 2> = a * b;
    // [1 2] * [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
    // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
    assert_eq!(c.get(0, 0), 19);
    assert_eq!(c.get(0, 1), 22);
    assert_eq!(c.get(1, 0), 43);
    assert_eq!(c.get(1, 1), 50);
}

#[test]
fn mul_rectangular_matrices() {
    let a: ConstMatrix<i32, 2, 3> = ConstMatrix::new(vec![1, 2, 3, 4, 5, 6]);
    let b: ConstMatrix<i32, 3, 2> = ConstMatrix::new(vec![7, 8, 9, 10, 11, 12]);
    let c: ConstMatrix<i32, 2, 2> = a * b;
    // [1 2 3] * [7  8 ] = [1*7+2*9+3*11   1*8+2*10+3*12] = [58  64]
    // [4 5 6]   [9  10]   [4*7+5*9+6*11   4*8+5*10+6*12]   [139 154]
    //           [11 12]
    assert_eq!(c.get(0, 0), 58);
    assert_eq!(c.get(0, 1), 64);
    assert_eq!(c.get(1, 0), 139);
    assert_eq!(c.get(1, 1), 154);
}

#[test]
fn mul_identity() {
    let a: ConstMatrix<i32, 3, 3> = ConstMatrix::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let identity: ConstMatrix<i32, 3, 3> = ConstMatrix::identity();
    let c: ConstMatrix<i32, 3, 3> = &a * &identity;
    assert_eq!(c.get(0, 0), 1);
    assert_eq!(c.get(0, 1), 2);
    assert_eq!(c.get(1, 1), 5);
    assert_eq!(c.get(2, 2), 9);
}

#[test]
fn mul_owned_ref() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let b: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![5, 6, 7, 8]);
    let c: ConstMatrix<i32, 2, 2> = a * &b;
    assert_eq!(c.get(0, 0), 19);
    assert_eq!(c.get(1, 1), 50);
    // b should still be usable
    assert_eq!(b.get(0, 0), 5);
}

#[test]
fn mul_ref_ref() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let b: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![5, 6, 7, 8]);
    let c: ConstMatrix<i32, 2, 2> = &a * &b;
    assert_eq!(c.get(0, 0), 19);
    assert_eq!(c.get(1, 1), 50);
    // Both should still be usable
    assert_eq!(a.get(0, 0), 1);
    assert_eq!(b.get(0, 0), 5);
}

#[test]
fn mul_f64() {
    let a: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0]);
    let b: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![0.5, 0.5, 0.5, 0.5]);
    let c: ConstMatrix<f64, 2, 2> = a * b;
    assert_eq!(c.get(0, 0), 1.5);
    assert_eq!(c.get(0, 1), 1.5);
    assert_eq!(c.get(1, 0), 3.5);
    assert_eq!(c.get(1, 1), 3.5);
}

#[test]
fn mul_complex() {
    use num_complex::Complex;
    let a: ConstMatrix<Complex<f64>, 2, 2> = ConstMatrix::new(vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(0.0, -1.0),
        Complex::new(1.0, 0.0),
    ]);
    let b: ConstMatrix<Complex<f64>, 2, 2> = ConstMatrix::new(vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
    ]);
    let c: ConstMatrix<Complex<f64>, 2, 2> = a * b;
    assert_eq!(c.get(0, 0), Complex::new(1.0, 0.0));
    assert_eq!(c.get(0, 1), Complex::new(0.0, 1.0));
}

// Scalar multiplication tests
#[test]
fn scalar_mul_matrix_owned() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let c = a * 5;
    assert_eq!(c.get(0, 0), 5);
    assert_eq!(c.get(0, 1), 10);
    assert_eq!(c.get(1, 0), 15);
    assert_eq!(c.get(1, 1), 20);
}

#[test]
fn scalar_mul_matrix_ref() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let c = &a * 5;
    assert_eq!(c.get(0, 0), 5);
    assert_eq!(c.get(1, 1), 20);
    // a should still be usable
    assert_eq!(a.get(0, 0), 1);
}

#[test]
fn scalar_mul_left_f64() {
    let a: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0]);
    let c = 2.5 * a;
    assert_eq!(c.get(0, 0), 2.5);
    assert_eq!(c.get(0, 1), 5.0);
    assert_eq!(c.get(1, 0), 7.5);
    assert_eq!(c.get(1, 1), 10.0);
}

#[test]
fn scalar_mul_left_f64_ref() {
    let a: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0]);
    let c = 2.5 * &a;
    assert_eq!(c.get(0, 0), 2.5);
    assert_eq!(c.get(1, 1), 10.0);
    // a should still be usable
    assert_eq!(a.get(0, 0), 1.0);
}

#[test]
fn scalar_mul_left_f32() {
    let a: ConstMatrix<f32, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0]);
    let c = 2.0_f32 * a;
    assert_eq!(c.get(0, 0), 2.0);
    assert_eq!(c.get(1, 1), 8.0);
}

#[test]
fn scalar_mul_left_i32() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let c = 3 * a;
    assert_eq!(c.get(0, 0), 3);
    assert_eq!(c.get(1, 1), 12);
}

#[test]
fn scalar_mul_commutative() {
    let a: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0]);
    let c1 = &a * 2.0;
    let c2 = 2.0 * &a;
    assert_eq!(c1.get(0, 0), c2.get(0, 0));
    assert_eq!(c1.get(0, 1), c2.get(0, 1));
    assert_eq!(c1.get(1, 0), c2.get(1, 0));
    assert_eq!(c1.get(1, 1), c2.get(1, 1));
}

#[test]
fn scalar_mul_zero() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let c = a * 0;
    assert_eq!(c.get(0, 0), 0);
    assert_eq!(c.get(0, 1), 0);
    assert_eq!(c.get(1, 0), 0);
    assert_eq!(c.get(1, 1), 0);
}

#[test]
fn scalar_mul_negative() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let c = a * (-1);
    assert_eq!(c.get(0, 0), -1);
    assert_eq!(c.get(0, 1), -2);
    assert_eq!(c.get(1, 0), -3);
    assert_eq!(c.get(1, 1), -4);
}

// ============================================================================
// Additional Operations Tests (Phase 3)
// ============================================================================

// Trace tests
#[test]
fn trace_square_matrix() {
    let m: ConstMatrix<i32, 3, 3> = ConstMatrix::new(vec![
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    ]);
    assert_eq!(m.trace(), 15); // 1 + 5 + 9
}

#[test]
fn trace_identity() {
    let m: ConstMatrix<i32, 4, 4> = ConstMatrix::identity();
    assert_eq!(m.trace(), 4);
}

#[test]
fn trace_zero_matrix() {
    let m: ConstMatrix<i32, 3, 3> = ConstMatrix::zero();
    assert_eq!(m.trace(), 0);
}

#[test]
fn trace_rectangular() {
    // For non-square, trace sums min(R, C) diagonal elements
    let m: ConstMatrix<i32, 2, 3> = ConstMatrix::new(vec![
        1, 2, 3,
        4, 5, 6,
    ]);
    assert_eq!(m.trace(), 6); // 1 + 5 (min(2,3) = 2 elements)
}

#[test]
fn trace_f64() {
    let m: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.5, 2.5, 3.5, 4.5]);
    assert_eq!(m.trace(), 6.0); // 1.5 + 4.5
}

// Dot product tests
#[test]
fn dot_product_basic() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let b: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![5, 6, 7, 8]);
    assert_eq!(a.dot(&b), 70); // 1*5 + 2*6 + 3*7 + 4*8
}

#[test]
fn dot_product_vectors() {
    let a: ConstMatrix<i32, 3, 1> = ConstMatrix::new(vec![1, 2, 3]);
    let b: ConstMatrix<i32, 3, 1> = ConstMatrix::new(vec![4, 5, 6]);
    assert_eq!(a.dot(&b), 32); // 1*4 + 2*5 + 3*6
}

#[test]
fn dot_product_f64() {
    let a: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0]);
    let b: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![0.5, 0.5, 0.5, 0.5]);
    assert_eq!(a.dot(&b), 5.0); // 0.5 + 1.0 + 1.5 + 2.0
}

#[test]
fn dot_product_with_self() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    assert_eq!(a.dot(&a), 30); // 1 + 4 + 9 + 16
}

#[test]
fn dot_product_zero() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let zero: ConstMatrix<i32, 2, 2> = ConstMatrix::zero();
    assert_eq!(a.dot(&zero), 0);
}

// Outer product tests
#[test]
fn outer_product_basic() {
    let a: ConstMatrix<i32, 2, 1> = ConstMatrix::new(vec![1, 2]);
    let b: ConstMatrix<i32, 3, 1> = ConstMatrix::new(vec![3, 4, 5]);
    let c: Matrix<i32> = a.outer(&b);
    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 3);
    assert_eq!(c.get(0, 0), 3);  // 1 * 3
    assert_eq!(c.get(0, 1), 4);  // 1 * 4
    assert_eq!(c.get(0, 2), 5);  // 1 * 5
    assert_eq!(c.get(1, 0), 6);  // 2 * 3
    assert_eq!(c.get(1, 1), 8);  // 2 * 4
    assert_eq!(c.get(1, 2), 10); // 2 * 5
}

#[test]
fn outer_product_vectors() {
    let a: ConstMatrix<i32, 3, 1> = ConstMatrix::new(vec![1, 2, 3]);
    let b: ConstMatrix<i32, 2, 1> = ConstMatrix::new(vec![4, 5]);
    let c: Matrix<i32> = a.outer(&b);
    assert_eq!(c.rows(), 3);
    assert_eq!(c.cols(), 2);
    assert_eq!(c.get(0, 0), 4);  // 1 * 4
    assert_eq!(c.get(0, 1), 5);  // 1 * 5
    assert_eq!(c.get(1, 0), 8);  // 2 * 4
    assert_eq!(c.get(2, 1), 15); // 3 * 5
}

#[test]
fn outer_product_f64() {
    let a: ConstMatrix<f64, 2, 1> = ConstMatrix::new(vec![1.5, 2.5]);
    let b: ConstMatrix<f64, 2, 1> = ConstMatrix::new(vec![2.0, 3.0]);
    let c: Matrix<f64> = a.outer(&b);
    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 2);
    assert_eq!(c.get(0, 0), 3.0);  // 1.5 * 2.0
    assert_eq!(c.get(0, 1), 4.5);  // 1.5 * 3.0
    assert_eq!(c.get(1, 0), 5.0);  // 2.5 * 2.0
    assert_eq!(c.get(1, 1), 7.5);  // 2.5 * 3.0
}

// Matrix power tests
#[test]
fn pow_zero() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let result: ConstMatrix<i32, 2, 2> = a.pow(0);
    // A^0 = Identity
    assert_eq!(result.get(0, 0), 1);
    assert_eq!(result.get(0, 1), 0);
    assert_eq!(result.get(1, 0), 0);
    assert_eq!(result.get(1, 1), 1);
}

#[test]
fn pow_one() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let result: ConstMatrix<i32, 2, 2> = a.pow(1);
    assert_eq!(result.get(0, 0), 1);
    assert_eq!(result.get(0, 1), 2);
    assert_eq!(result.get(1, 0), 3);
    assert_eq!(result.get(1, 1), 4);
}

#[test]
fn pow_two() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let result: ConstMatrix<i32, 2, 2> = a.pow(2);
    // [1 2] * [1 2] = [7  10]
    // [3 4]   [3 4]   [15 22]
    assert_eq!(result.get(0, 0), 7);
    assert_eq!(result.get(0, 1), 10);
    assert_eq!(result.get(1, 0), 15);
    assert_eq!(result.get(1, 1), 22);
}

#[test]
fn pow_three() {
    let a: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    let result: ConstMatrix<i32, 2, 2> = a.pow(3);
    // A^3 = A * A * A
    let a_squared = &a * &a;
    let expected: ConstMatrix<i32, 2, 2> = &a_squared * &a;
    assert_eq!(result.get(0, 0), expected.get(0, 0));
    assert_eq!(result.get(1, 1), expected.get(1, 1));
}

#[test]
fn pow_identity() {
    let identity: ConstMatrix<i32, 3, 3> = ConstMatrix::identity();
    let result: ConstMatrix<i32, 3, 3> = identity.pow(5);
    // Identity to any power is still identity
    assert_eq!(result.get(0, 0), 1);
    assert_eq!(result.get(1, 1), 1);
    assert_eq!(result.get(2, 2), 1);
    assert_eq!(result.get(0, 1), 0);
}

#[test]
fn pow_f64() {
    let a: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0]);
    let result: ConstMatrix<f64, 2, 2> = a.pow(2);
    assert_eq!(result.get(0, 0), 7.0);
    assert_eq!(result.get(1, 1), 22.0);
}

// Cross product tests
#[test]
fn cross_product_col_vectors() {
    let a: ConstMatrix<i32, 3, 1> = ConstMatrix::new(vec![1, 0, 0]);
    let b: ConstMatrix<i32, 3, 1> = ConstMatrix::new(vec![0, 1, 0]);
    let c = a.cross(&b);
    assert_eq!(c.get(0, 0), 0);
    assert_eq!(c.get(1, 0), 0);
    assert_eq!(c.get(2, 0), 1); // i × j = k
}

#[test]
fn cross_product_row_vectors() {
    let a: ConstMatrix<i32, 1, 3> = ConstMatrix::new(vec![1, 0, 0]);
    let b: ConstMatrix<i32, 1, 3> = ConstMatrix::new(vec![0, 1, 0]);
    let c = a.cross(&b);
    assert_eq!(c.get(0, 0), 0);
    assert_eq!(c.get(0, 1), 0);
    assert_eq!(c.get(0, 2), 1); // i × j = k
}

#[test]
fn cross_product_anti_commutative() {
    let a: ConstMatrix<i32, 3, 1> = ConstMatrix::new(vec![1, 2, 3]);
    let b: ConstMatrix<i32, 3, 1> = ConstMatrix::new(vec![4, 5, 6]);
    let c1 = a.cross(&b);
    let c2 = b.cross(&a);
    // a × b = -(b × a)
    assert_eq!(c1.get(0, 0), -c2.get(0, 0));
    assert_eq!(c1.get(1, 0), -c2.get(1, 0));
    assert_eq!(c1.get(2, 0), -c2.get(2, 0));
}

#[test]
fn cross_product_parallel_vectors() {
    let a: ConstMatrix<i32, 3, 1> = ConstMatrix::new(vec![1, 2, 3]);
    let b: ConstMatrix<i32, 3, 1> = ConstMatrix::new(vec![2, 4, 6]); // b = 2*a
    let c = a.cross(&b);
    // Parallel vectors have zero cross product
    assert_eq!(c.get(0, 0), 0);
    assert_eq!(c.get(1, 0), 0);
    assert_eq!(c.get(2, 0), 0);
}

#[test]
fn cross_product_with_self() {
    let a: ConstMatrix<i32, 3, 1> = ConstMatrix::new(vec![1, 2, 3]);
    let c = a.cross(&a);
    // a × a = 0
    assert_eq!(c.get(0, 0), 0);
    assert_eq!(c.get(1, 0), 0);
    assert_eq!(c.get(2, 0), 0);
}

#[test]
fn cross_product_basis_vectors() {
    // Test right-hand rule: i × j = k, j × k = i, k × i = j
    let i: ConstMatrix<i32, 3, 1> = ConstMatrix::new(vec![1, 0, 0]);
    let j: ConstMatrix<i32, 3, 1> = ConstMatrix::new(vec![0, 1, 0]);
    let k: ConstMatrix<i32, 3, 1> = ConstMatrix::new(vec![0, 0, 1]);

    let ij = i.cross(&j);
    assert_eq!(ij.get(0, 0), 0);
    assert_eq!(ij.get(1, 0), 0);
    assert_eq!(ij.get(2, 0), 1); // k

    let jk = j.cross(&k);
    assert_eq!(jk.get(0, 0), 1); // i
    assert_eq!(jk.get(1, 0), 0);
    assert_eq!(jk.get(2, 0), 0);

    let ki = k.cross(&i);
    assert_eq!(ki.get(0, 0), 0);
    assert_eq!(ki.get(1, 0), 1); // j
    assert_eq!(ki.get(2, 0), 0);
}

#[test]
fn cross_product_f64() {
    let a: ConstMatrix<f64, 3, 1> = ConstMatrix::new(vec![1.0, 2.0, 3.0]);
    let b: ConstMatrix<f64, 3, 1> = ConstMatrix::new(vec![4.0, 5.0, 6.0]);
    let c = a.cross(&b);
    // [1,2,3] × [4,5,6] = [2*6-3*5, 3*4-1*6, 1*5-2*4] = [-3, 6, -3]
    assert_eq!(c.get(0, 0), -3.0);
    assert_eq!(c.get(1, 0), 6.0);
    assert_eq!(c.get(2, 0), -3.0);
}

// Approximate equality tests
#[test]
fn approx_eq_exact_match() {
    let a: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0]);
    let b: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0]);
    assert!(a.approx_eq(&b, 1e-10));
}

#[test]
fn approx_eq_within_tolerance() {
    let a: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0]);
    let b: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0001, 2.0001, 3.0001, 4.0001]);
    assert!(a.approx_eq(&b, 0.001));
    assert!(a.approx_eq(&b, 0.00015)); // Slightly larger to account for floating point
}

#[test]
fn approx_eq_outside_tolerance() {
    let a: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0]);
    let b: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0001, 2.0001, 3.0001, 4.0001]);
    assert!(!a.approx_eq(&b, 0.00001));
}

#[test]
fn approx_eq_f32() {
    let a: ConstMatrix<f32, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 3.0, 4.0]);
    let b: ConstMatrix<f32, 2, 2> = ConstMatrix::new(vec![1.001, 2.001, 3.001, 4.001]);
    assert!(a.approx_eq(&b, 0.01));
    assert!(!a.approx_eq(&b, 0.0001));
}

#[test]
fn approx_eq_complex() {
    use num_complex::Complex;
    let a: ConstMatrix<Complex<f64>, 2, 2> = ConstMatrix::new(vec![
        Complex::new(1.0, 2.0),
        Complex::new(3.0, 4.0),
        Complex::new(5.0, 6.0),
        Complex::new(7.0, 8.0),
    ]);
    let b: ConstMatrix<Complex<f64>, 2, 2> = ConstMatrix::new(vec![
        Complex::new(1.0001, 2.0001),
        Complex::new(3.0001, 4.0001),
        Complex::new(5.0001, 6.0001),
        Complex::new(7.0001, 8.0001),
    ]);
    assert!(a.approx_eq(&b, 0.001));
}

// ============================================================================
// Phase 4 Tests: Norms, Determinants, Inverses
// ============================================================================

// Norm tests
#[test]
fn frobenius_norm_2x2() {
    let m: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![3.0, 4.0, 0.0, 0.0]);
    assert_eq!(m.frobenius_norm(), 5.0); // sqrt(9 + 16) = 5
}

#[test]
fn frobenius_norm_identity() {
    let m: ConstMatrix<f64, 3, 3> = ConstMatrix::identity();
    assert!((m.frobenius_norm() - 1.732050807568877).abs() < 1e-10); // sqrt(3)
}

#[test]
fn inf_norm_2x2() {
    let m: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, -2, 3, -4]);
    assert_eq!(m.inf_norm(), 7); // max(|1| + |-2|, |3| + |-4|) = max(3, 7) = 7
}

#[test]
fn one_norm_2x2() {
    let m: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, -2, 3, -4]);
    assert_eq!(m.one_norm(), 6); // max(|1| + |3|, |-2| + |-4|) = max(4, 6) = 6
}

#[test]
fn norms_on_3x3() {
    let m: ConstMatrix<i32, 3, 3> = ConstMatrix::new(vec![
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    ]);
    assert_eq!(m.inf_norm(), 24); // max row sum: 7+8+9 = 24
    assert_eq!(m.one_norm(), 18); // max col sum: 3+6+9 = 18
}

// 2x2 determinant tests
#[test]
fn det_2x2_basic() {
    let m: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 3, 4]);
    assert_eq!(m.determinant(), -2); // 1*4 - 2*3 = -2
}

#[test]
fn det_2x2_identity() {
    let m: ConstMatrix<i32, 2, 2> = ConstMatrix::identity();
    assert_eq!(m.determinant(), 1);
}

#[test]
fn det_2x2_singular() {
    let m: ConstMatrix<i32, 2, 2> = ConstMatrix::new(vec![1, 2, 2, 4]);
    assert_eq!(m.determinant(), 0); // Singular (row 2 = 2 * row 1)
}

#[test]
fn det_2x2_f64() {
    let m: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![4.0, 7.0, 2.0, 6.0]);
    assert_eq!(m.determinant(), 10.0); // 4*6 - 7*2 = 24 - 14 = 10
}

// 2x2 inverse tests
#[test]
fn inverse_2x2_basic() {
    let m: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![4.0, 7.0, 2.0, 6.0]);
    let inv = m.inverse().unwrap();

    // Check specific values
    assert!((inv.get(0, 0) - 0.6).abs() < 1e-10);
    assert!((inv.get(0, 1) - (-0.7)).abs() < 1e-10);
    assert!((inv.get(1, 0) - (-0.2)).abs() < 1e-10);
    assert!((inv.get(1, 1) - 0.4).abs() < 1e-10);

    // Verify M * M^-1 = I
    let identity = &m * &inv;
    assert!(identity.approx_eq(&ConstMatrix::identity(), 1e-10));
}

#[test]
fn inverse_2x2_identity() {
    let m: ConstMatrix<f64, 2, 2> = ConstMatrix::identity();
    let inv = m.inverse().unwrap();
    assert!(inv.approx_eq(&ConstMatrix::identity(), 1e-10));
}

#[test]
fn inverse_2x2_singular_returns_none() {
    let m: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![1.0, 2.0, 2.0, 4.0]);
    assert!(m.inverse().is_none()); // Singular matrix
}

// 3x3 determinant tests
#[test]
fn det_3x3_basic() {
    let m: ConstMatrix<i32, 3, 3> = ConstMatrix::new(vec![
        1, 2, 3,
        0, 1, 4,
        5, 6, 0,
    ]);
    assert_eq!(m.determinant(), 1);
}

#[test]
fn det_3x3_identity() {
    let m: ConstMatrix<i32, 3, 3> = ConstMatrix::identity();
    assert_eq!(m.determinant(), 1);
}

#[test]
fn det_3x3_singular() {
    let m: ConstMatrix<i32, 3, 3> = ConstMatrix::new(vec![
        1, 2, 3,
        2, 4, 6,
        3, 6, 9,
    ]);
    assert_eq!(m.determinant(), 0); // All rows are multiples
}

#[test]
fn det_3x3_various() {
    let m: ConstMatrix<i32, 3, 3> = ConstMatrix::new(vec![
        6, 1, 1,
        4, -2, 5,
        2, 8, 7,
    ]);
    assert_eq!(m.determinant(), -306);
}

// 3x3 inverse tests
#[test]
fn inverse_3x3_basic() {
    let m: ConstMatrix<f64, 3, 3> = ConstMatrix::new(vec![
        1.0, 2.0, 3.0,
        0.0, 1.0, 4.0,
        5.0, 6.0, 0.0,
    ]);
    let inv = m.inverse().unwrap();

    // Verify M * M^-1 ≈ I
    let identity = &m * &inv;
    assert!(identity.approx_eq(&ConstMatrix::identity(), 1e-10));

    // Verify M^-1 * M ≈ I
    let identity2 = &inv * &m;
    assert!(identity2.approx_eq(&ConstMatrix::identity(), 1e-10));
}

#[test]
fn inverse_3x3_identity() {
    let m: ConstMatrix<f64, 3, 3> = ConstMatrix::identity();
    let inv = m.inverse().unwrap();
    assert!(inv.approx_eq(&ConstMatrix::identity(), 1e-10));
}

#[test]
fn inverse_3x3_singular_returns_none() {
    let m: ConstMatrix<f64, 3, 3> = ConstMatrix::new(vec![
        1.0, 2.0, 3.0,
        2.0, 4.0, 6.0,
        3.0, 6.0, 9.0,
    ]);
    assert!(m.inverse().is_none()); // Singular matrix
}

#[test]
fn inverse_3x3_known_matrix() {
    let m: ConstMatrix<f64, 3, 3> = ConstMatrix::new(vec![
        3.0, 0.0, 2.0,
        2.0, 0.0, -2.0,
        0.0, 1.0, 1.0,
    ]);
    let inv = m.inverse().unwrap();

    // Check that det = 10
    assert!((m.determinant() - 10.0).abs() < 1e-10);

    // Verify M * M^-1 = I
    let identity = &m * &inv;
    assert!(identity.approx_eq(&ConstMatrix::identity(), 1e-10));
}

// 4x4 determinant tests
#[test]
fn det_4x4_identity() {
    let m: ConstMatrix<i32, 4, 4> = ConstMatrix::identity();
    assert_eq!(m.determinant(), 1);
}

#[test]
fn det_4x4_basic() {
    let m: ConstMatrix<i32, 4, 4> = ConstMatrix::new(vec![
        1, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 3, 0,
        0, 0, 0, 4,
    ]);
    assert_eq!(m.determinant(), 24); // Product of diagonal = 1*2*3*4 = 24
}

#[test]
fn det_4x4_singular() {
    let m: ConstMatrix<i32, 4, 4> = ConstMatrix::new(vec![
        1, 2, 3, 4,
        2, 4, 6, 8,
        5, 6, 7, 8,
        9, 10, 11, 12,
    ]);
    assert_eq!(m.determinant(), 0); // Row 2 = 2 * Row 1
}

#[test]
fn det_4x4_known_value() {
    // Use a matrix with a known non-zero determinant
    let m: ConstMatrix<i32, 4, 4> = ConstMatrix::new(vec![
        1, 0, 2, -1,
        3, 0, 0, 5,
        2, 1, 4, -3,
        1, 0, 5, 0,
    ]);
    assert_eq!(m.determinant(), 30);
}

// 4x4 inverse tests
#[test]
fn inverse_4x4_identity() {
    let m: ConstMatrix<f64, 4, 4> = ConstMatrix::identity();
    let inv = m.inverse().unwrap();
    assert!(inv.approx_eq(&ConstMatrix::identity(), 1e-10));
}

#[test]
fn inverse_4x4_diagonal() {
    let m: ConstMatrix<f64, 4, 4> = ConstMatrix::new(vec![
        2.0, 0.0, 0.0, 0.0,
        0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 4.0, 0.0,
        0.0, 0.0, 0.0, 5.0,
    ]);
    let inv = m.inverse().unwrap();

    // For diagonal matrix, inverse is 1/diagonal
    assert!((inv.get(0, 0) - 0.5).abs() < 1e-10);
    assert!((inv.get(1, 1) - 1.0/3.0).abs() < 1e-10);
    assert!((inv.get(2, 2) - 0.25).abs() < 1e-10);
    assert!((inv.get(3, 3) - 0.2).abs() < 1e-10);

    // Verify M * M^-1 = I
    let identity = &m * &inv;
    assert!(identity.approx_eq(&ConstMatrix::identity(), 1e-10));
}

#[test]
fn inverse_4x4_full_matrix() {
    // Use a matrix with known non-zero determinant
    let m: ConstMatrix<f64, 4, 4> = ConstMatrix::new(vec![
        1.0, 0.0, 2.0, -1.0,
        3.0, 0.0, 0.0, 5.0,
        2.0, 1.0, 4.0, -3.0,
        1.0, 0.0, 5.0, 0.0,
    ]);
    let inv = m.inverse().unwrap();

    // Verify M * M^-1 ≈ I
    let identity = &m * &inv;
    assert!(identity.approx_eq(&ConstMatrix::identity(), 1e-9));

    // Verify M^-1 * M ≈ I
    let identity2 = &inv * &m;
    assert!(identity2.approx_eq(&ConstMatrix::identity(), 1e-9));
}

#[test]
fn inverse_4x4_singular_returns_none() {
    let m: ConstMatrix<f64, 4, 4> = ConstMatrix::new(vec![
        1.0, 2.0, 3.0, 4.0,
        2.0, 4.0, 6.0, 8.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
    ]);
    assert!(m.inverse().is_none()); // Singular matrix
}
