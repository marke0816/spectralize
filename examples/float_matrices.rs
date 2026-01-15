use spectralize::Matrix;

fn main() {
    println!("=== Float Matrix Examples (f32 and f64) ===\n");

    // ==================== Matrix Creation ====================
    println!("1. Matrix Creation");

    // f64 matrix from vector
    let a_f64 = Matrix::new(2, 3, vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ]);
    println!("f64 matrix A (2x3):");
    print_matrix(&a_f64);

    // f32 identity matrix
    let identity_f32 = Matrix::<f32>::identity(3, 3);
    println!("f32 identity matrix (3x3):");
    print_matrix(&identity_f32);

    // Zero matrix
    let zero_f64 = Matrix::<f64>::zero(2, 2);
    println!("f64 zero matrix (2x2):");
    print_matrix(&zero_f64);

    // ==================== Element Access ====================
    println!("\n2. Element Access");

    let element = a_f64.get(0, 1);
    println!("Element at (0, 1): {}", element);

    // Checked access (returns Result)
    match a_f64.try_get(0, 1) {
        Ok(val) => println!("Safe access at (0, 1): {}", val),
        Err(e) => println!("Error: {:?}", e),
    }

    // Row access (zero-copy slice)
    let row = a_f64.row(0);
    println!("First row: {:?}", row);

    // Column iteration without allocation
    let col: Vec<f64> = a_f64.col_iter(1).copied().collect();
    println!("Second column: {:?}", col);

    // ==================== Matrix Modification ====================
    println!("\n3. Matrix Modification");

    let mut b_f64 = Matrix::<f64>::zero(2, 2);
    b_f64.set(0, 0, 1.0);
    b_f64.set(0, 1, 2.0);
    b_f64.set(1, 0, 3.0);
    b_f64.set(1, 1, 4.0);
    println!("Modified matrix B:");
    print_matrix(&b_f64);

    // ==================== Transpose ====================
    println!("\n4. Transpose");

    let transposed = a_f64.transpose();
    println!("Transpose of A (3x2):");
    print_matrix(&transposed);

    // ==================== Arithmetic Operations ====================
    println!("\n5. Arithmetic Operations");

    let c_f64 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let d_f64 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

    println!("Matrix C:");
    print_matrix(&c_f64);
    println!("Matrix D:");
    print_matrix(&d_f64);

    // Addition
    let sum = &c_f64 + &d_f64;
    println!("C + D:");
    print_matrix(&sum);

    // Subtraction
    let diff = &c_f64 - &d_f64;
    println!("C - D:");
    print_matrix(&diff);

    // Matrix multiplication
    let product = &c_f64 * &d_f64;
    println!("C * D:");
    print_matrix(&product);

    // Scalar multiplication
    let scaled = &c_f64 * 2.0;
    println!("C * 2:");
    print_matrix(&scaled);

    // Scalar multiplication (scalar on left)
    let scaled_left = 3.0 * &c_f64;
    println!("3 * C:");
    print_matrix(&scaled_left);

    // ==================== Trace ====================
    println!("\n6. Trace (sum of diagonal)");

    let trace_c = c_f64.trace();
    println!("Trace of C: {}", trace_c);

    // ==================== Matrix Power ====================
    println!("\n7. Matrix Power");

    let c_squared = c_f64.pow(2);
    println!("C^2:");
    print_matrix(&c_squared);

    let c_cubed = c_f64.pow(3);
    println!("C^3:");
    print_matrix(&c_cubed);

    // ==================== Determinant ====================
    println!("\n8. Determinant");

    let det_c = c_f64.determinant();
    println!("Determinant of C: {}", det_c);

    // Determinant with custom tolerance
    let det_with_tol = c_f64.determinant_with_tol(1e-10);
    println!("Determinant of C (tol=1e-10): {}", det_with_tol);

    // ==================== Invertibility ====================
    println!("\n9. Invertibility Check");

    println!("Is C invertible? {}", c_f64.is_invertible());
    println!("Is zero matrix invertible? {}", zero_f64.is_invertible());

    // Invertibility with custom tolerance
    let nearly_singular = Matrix::new(2, 2, vec![
        1.0, 2.0,
        1.0, 2.0000001,
    ]);
    println!("Nearly singular matrix:");
    print_matrix(&nearly_singular);
    println!("  Invertible (default tol)? {}", nearly_singular.is_invertible());
    println!("  Invertible (tol=1e-3)? {}", nearly_singular.is_invertible_with_tol(1e-3));

    // ==================== Matrix Norms ====================
    println!("\n10. Matrix Norms");

    let e_f64 = Matrix::new(2, 3, vec![
        3.0, 4.0, 0.0,
        0.0, 0.0, 5.0,
    ]);
    println!("Matrix E:");
    print_matrix(&e_f64);

    println!("Frobenius norm: {:.4}", e_f64.norm_fro());
    println!("1-norm (max column sum): {:.4}", e_f64.norm_one());
    println!("Infinity norm (max row sum): {:.4}", e_f64.norm_inf());

    // ==================== Approximate Equality ====================
    println!("\n11. Approximate Equality");

    let f1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let f2 = Matrix::new(2, 2, vec![1.0000001, 2.0, 3.0, 4.0]);

    println!("F1 == F2 (exact)? {}", f1 == f2);
    println!("F1 ≈ F2 (tol=1e-3)? {}", f1.approx_eq(&f2, 1e-3));
    println!("F1 ≈ F2 (tol=1e-9)? {}", f1.approx_eq(&f2, 1e-9));

    // ==================== Concatenation ====================
    println!("\n12. Matrix Concatenation");

    let left = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let right = Matrix::new(2, 1, vec![5.0, 6.0]);

    println!("Left matrix:");
    print_matrix(&left);
    println!("Right matrix:");
    print_matrix(&right);

    let h_concat = left.with_cols(&right);
    println!("Horizontal concatenation:");
    print_matrix(&h_concat);

    let top = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let bottom = Matrix::new(1, 2, vec![5.0, 6.0]);

    let v_concat = top.with_rows(&bottom);
    println!("Vertical concatenation:");
    print_matrix(&v_concat);

    // ==================== Checked APIs ====================
    println!("\n13. Checked APIs (error handling)");

    let g = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Safe get
    match g.try_get(0, 1) {
        Ok(val) => println!("try_get(0, 1): {}", val),
        Err(e) => println!("Error: {:?}", e),
    }

    match g.try_get(5, 5) {
        Ok(val) => println!("try_get(5, 5): {}", val),
        Err(e) => println!("Error: {:?}", e),
    }

    // Safe trace (requires square matrix)
    match g.try_trace() {
        Ok(val) => println!("Trace: {}", val),
        Err(e) => println!("Cannot compute trace: {:?}", e),
    }

    // Safe determinant
    match g.try_determinant() {
        Ok(val) => println!("Determinant: {}", val),
        Err(e) => println!("Cannot compute determinant: {:?}", e),
    }

    // ==================== f32 Examples ====================
    println!("\n14. f32 Examples (32-bit floats)");

    let a_f32 = Matrix::new(2, 2, vec![
        1.0f32, 2.0f32,
        3.0f32, 4.0f32,
    ]);
    let b_f32 = Matrix::new(2, 2, vec![
        5.0f32, 6.0f32,
        7.0f32, 8.0f32,
    ]);

    println!("f32 matrix A:");
    print_matrix(&a_f32);
    println!("f32 matrix B:");
    print_matrix(&b_f32);

    let sum_f32 = &a_f32 + &b_f32;
    println!("A + B (f32):");
    print_matrix(&sum_f32);

    println!("Determinant of A (f32): {}", a_f32.determinant());
    println!("Frobenius norm of A (f32): {}", a_f32.norm_fro());

    // f32 uses larger default tolerance (1e-6 vs 1e-12 for f64)
    println!("Is A invertible (f32)? {}", a_f32.is_invertible());

    println!("\n=== All float examples completed! ===");
}

fn print_matrix<T: std::fmt::Debug + spectralize::MatrixElement>(matrix: &Matrix<T>) {
    for row in 0..matrix.rows() {
        print!("  ");
        for col in 0..matrix.cols() {
            print!("{:?} ", matrix.get_ref(row, col));
        }
        println!();
    }
}
