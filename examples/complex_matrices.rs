use num_complex::Complex;
use spectralize::Matrix;

fn main() {
    println!("=== Complex Number Matrix Examples ===\n");

    // ==================== Matrix Creation ====================
    println!("1. Matrix Creation");

    // Complex<f64> matrix
    let a = Matrix::new(2, 3, vec![
        Complex::new(1.0, 1.0),  // 1 + i
        Complex::new(2.0, 0.0),  // 2
        Complex::new(0.0, 3.0),  // 3i
        Complex::new(4.0, -1.0), // 4 - i
        Complex::new(5.0, 2.0),  // 5 + 2i
        Complex::new(1.0, 1.0),  // 1 + i
    ]);
    println!("Complex matrix A (2x3):");
    print_matrix(&a);

    // Identity matrix
    let identity = Matrix::<Complex<f64>>::identity(3, 3);
    println!("Complex identity matrix (3x3):");
    print_matrix(&identity);

    // Zero matrix
    let zero = Matrix::<Complex<f64>>::zero(2, 2);
    println!("Complex zero matrix (2x2):");
    print_matrix(&zero);

    // ==================== Element Access ====================
    println!("\n2. Element Access");

    let element = a.get(0, 1);
    println!("Element at (0, 1): {}", element);

    // Checked access
    match a.try_get(0, 1) {
        Ok(val) => println!("Safe access at (0, 1): {}", val),
        Err(e) => println!("Error: {:?}", e),
    }

    // Row access
    let row = a.row(0);
    println!("First row: {:?}", row);

    // Column iteration
    let col: Vec<Complex<f64>> = a.col_iter(1).copied().collect();
    println!("Second column: {:?}", col);

    // ==================== Matrix Modification ====================
    println!("\n3. Matrix Modification");

    let mut b = Matrix::<Complex<f64>>::zero(2, 2);
    b.set(0, 0, Complex::new(1.0, 0.0));
    b.set(0, 1, Complex::new(0.0, 1.0));  // i
    b.set(1, 0, Complex::new(-1.0, 0.0)); // -1
    b.set(1, 1, Complex::new(0.0, -1.0)); // -i
    println!("Modified matrix B:");
    print_matrix(&b);

    // ==================== Transpose ====================
    println!("\n4. Transpose");

    let transposed = a.transpose();
    println!("Transpose of A (3x2):");
    print_matrix(&transposed);

    // ==================== Arithmetic Operations ====================
    println!("\n5. Arithmetic Operations");

    let c = Matrix::new(2, 2, vec![
        Complex::new(1.0, 1.0),
        Complex::new(2.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(3.0, 2.0),
    ]);
    let d = Matrix::new(2, 2, vec![
        Complex::new(1.0, -1.0),
        Complex::new(2.0, 1.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),
    ]);

    println!("Matrix C:");
    print_matrix(&c);
    println!("Matrix D:");
    print_matrix(&d);

    // Addition
    let sum = &c + &d;
    println!("C + D:");
    print_matrix(&sum);

    // Subtraction
    let diff = &c - &d;
    println!("C - D:");
    print_matrix(&diff);

    // Matrix multiplication
    let product = &c * &d;
    println!("C * D:");
    print_matrix(&product);

    // Scalar multiplication (real scalar)
    let scaled = &c * Complex::new(2.0, 0.0);
    println!("C * 2:");
    print_matrix(&scaled);

    // Complex scalar multiplication
    let scaled_complex = &c * Complex::new(0.0, 1.0); // multiply by i
    println!("C * i:");
    print_matrix(&scaled_complex);

    // ==================== Trace ====================
    println!("\n6. Trace (sum of diagonal)");

    let trace_c = c.trace();
    println!("Trace of C: {}", trace_c);

    // ==================== Matrix Power ====================
    println!("\n7. Matrix Power");

    let c_squared = c.pow(2);
    println!("C^2:");
    print_matrix(&c_squared);

    let c_cubed = c.pow(3);
    println!("C^3:");
    print_matrix(&c_cubed);

    // ==================== Determinant ====================
    println!("\n8. Determinant");

    let det_c = c.determinant();
    println!("Determinant of C: {}", det_c);

    // Interesting complex matrix
    let unitary_like = Matrix::new(2, 2, vec![
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),  // i
        Complex::new(0.0, -1.0), // -i
        Complex::new(1.0, 0.0),
    ]);
    println!("Unitary-like matrix:");
    print_matrix(&unitary_like);
    println!("Determinant: {}", unitary_like.determinant());

    // ==================== Invertibility ====================
    println!("\n9. Invertibility Check");

    println!("Is C invertible? {}", c.is_invertible());
    println!("Is zero matrix invertible? {}", zero.is_invertible());

    // Nearly singular complex matrix
    let nearly_singular = Matrix::new(2, 2, vec![
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(1.0, 0.0000001),
        Complex::new(2.0, 0.0),
    ]);
    println!("Nearly singular matrix:");
    print_matrix(&nearly_singular);
    println!("  Invertible (default tol)? {}", nearly_singular.is_invertible());
    println!("  Invertible (tol=1e-3)? {}", nearly_singular.is_invertible_with_tol(1e-3));

    // ==================== Matrix Norms ====================
    println!("\n10. Matrix Norms");

    let e = Matrix::new(2, 3, vec![
        Complex::new(3.0, 0.0),
        Complex::new(0.0, 4.0),  // 4i
        Complex::new(0.0, 0.0),
        Complex::new(5.0, 0.0),
        Complex::new(0.0, 0.0),
        Complex::new(3.0, 4.0),  // 3 + 4i (magnitude 5)
    ]);
    println!("Matrix E:");
    print_matrix(&e);

    // Note: norms return f64 for Complex<f64> matrices
    println!("Frobenius norm: {:.4}", e.norm_fro());
    println!("1-norm (max column sum): {:.4}", e.norm_one());
    println!("Infinity norm (max row sum): {:.4}", e.norm_inf());

    // ==================== Cross Product (3D Complex Vectors) ====================
    println!("\n11. Cross Product (3D Complex Vectors)");

    // Complex vectors
    let cv1 = Matrix::new(3, 1, vec![
        Complex::new(1.0, 0.0),  // 1
        Complex::new(0.0, 1.0),  // i
        Complex::new(0.0, 0.0),  // 0
    ]);
    let cv2 = Matrix::new(3, 1, vec![
        Complex::new(0.0, 0.0),  // 0
        Complex::new(1.0, 0.0),  // 1
        Complex::new(0.0, 1.0),  // i
    ]);

    println!("Complex vector v1 = (1, i, 0):");
    print_matrix(&cv1);
    println!("Complex vector v2 = (0, 1, i):");
    print_matrix(&cv2);

    let c_cross = cv1.cross(&cv2).unwrap();
    println!("v1 × v2:");
    print_matrix(&c_cross);
    println!("Formula: (i·i - 0·1, 0·0 - 1·i, 1·1 - i·0) = (-1, -i, 1)");

    // Verify orthogonality with complex dot product
    let cv1_dot_result = cv1.dot(&c_cross);
    let cv2_dot_result = cv2.dot(&c_cross);
    println!("\nOrthogonality check:");
    println!("v1 · (v1 × v2) = {} (magnitude: {:.10})",
             cv1_dot_result, cv1_dot_result.norm());
    println!("v2 · (v1 × v2) = {} (magnitude: {:.10})",
             cv2_dot_result, cv2_dot_result.norm());

    // Real-valued complex vectors
    let real_cv1 = Matrix::new(3, 1, vec![
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(4.0, 0.0),
    ]);
    let real_cv2 = Matrix::new(3, 1, vec![
        Complex::new(5.0, 0.0),
        Complex::new(6.0, 0.0),
        Complex::new(7.0, 0.0),
    ]);
    let real_cross = real_cv1.cross(&real_cv2).unwrap();
    println!("\nReal-valued complex vectors (2+0i, 3+0i, 4+0i) × (5+0i, 6+0i, 7+0i):");
    print_matrix(&real_cross);
    println!("Result: same as real arithmetic");

    // ==================== Approximate Equality ====================
    println!("\n12. Approximate Equality");

    let f1 = Matrix::new(2, 2, vec![
        Complex::new(1.0, 1.0),
        Complex::new(2.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(3.0, 2.0),
    ]);
    let f2 = Matrix::new(2, 2, vec![
        Complex::new(1.0000001, 1.0),
        Complex::new(2.0, 0.0000001),
        Complex::new(0.0, 1.0),
        Complex::new(3.0, 2.0),
    ]);

    println!("F1 == F2 (exact)? {}", f1 == f2);
    println!("F1 ≈ F2 (tol=1e-3)? {}", f1.approx_eq(&f2, 1e-3));
    println!("F1 ≈ F2 (tol=1e-9)? {}", f1.approx_eq(&f2, 1e-9));

    // ==================== Concatenation ====================
    println!("\n13. Matrix Concatenation");

    let left = Matrix::new(2, 2, vec![
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(4.0, 0.0),
    ]);
    let right = Matrix::new(2, 1, vec![
        Complex::new(0.0, 1.0),
        Complex::new(0.0, 2.0),
    ]);

    println!("Left matrix:");
    print_matrix(&left);
    println!("Right matrix:");
    print_matrix(&right);

    let h_concat = left.with_cols(&right);
    println!("Horizontal concatenation:");
    print_matrix(&h_concat);

    let top = Matrix::new(2, 2, vec![
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(4.0, 0.0),
    ]);
    let bottom = Matrix::new(1, 2, vec![
        Complex::new(5.0, 0.0),
        Complex::new(6.0, 0.0),
    ]);

    let v_concat = top.with_rows(&bottom);
    println!("Vertical concatenation:");
    print_matrix(&v_concat);

    // ==================== Checked APIs ====================
    println!("\n14. Checked APIs (error handling)");

    let g = Matrix::new(2, 3, vec![
        Complex::new(1.0, 1.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(4.0, -1.0),
        Complex::new(5.0, 0.0),
        Complex::new(6.0, 1.0),
    ]);

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

    // ==================== Complex<f32> Examples ====================
    println!("\n15. Complex<f32> Examples (32-bit floats)");

    let a_f32 = Matrix::new(2, 2, vec![
        Complex::new(1.0f32, 1.0f32),
        Complex::new(2.0f32, 0.0f32),
        Complex::new(0.0f32, 1.0f32),
        Complex::new(3.0f32, 2.0f32),
    ]);
    let b_f32 = Matrix::new(2, 2, vec![
        Complex::new(1.0f32, -1.0f32),
        Complex::new(2.0f32, 1.0f32),
        Complex::new(1.0f32, 0.0f32),
        Complex::new(0.0f32, 1.0f32),
    ]);

    println!("Complex<f32> matrix A:");
    print_matrix(&a_f32);
    println!("Complex<f32> matrix B:");
    print_matrix(&b_f32);

    let sum_f32 = &a_f32 + &b_f32;
    println!("A + B (Complex<f32>):");
    print_matrix(&sum_f32);

    println!("Determinant of A (Complex<f32>): {}", a_f32.determinant());
    println!("Frobenius norm of A (Complex<f32>): {}", a_f32.norm_fro());

    // Complex<f32> uses larger tolerance (f32 default: 1e-6)
    println!("Is A invertible (Complex<f32>)? {}", a_f32.is_invertible());

    // ==================== Special Complex Matrices ====================
    println!("\n16. Special Complex Matrices");

    // Pauli matrices (important in quantum mechanics)
    let pauli_x = Matrix::new(2, 2, vec![
        Complex::new(0.0, 0.0), Complex::new(1.0, 0.0),
        Complex::new(1.0, 0.0), Complex::new(0.0, 0.0),
    ]);
    println!("Pauli X matrix:");
    print_matrix(&pauli_x);
    println!("  Determinant: {}", pauli_x.determinant());

    let pauli_y = Matrix::new(2, 2, vec![
        Complex::new(0.0, 0.0),  Complex::new(0.0, -1.0),
        Complex::new(0.0, 1.0),  Complex::new(0.0, 0.0),
    ]);
    println!("Pauli Y matrix:");
    print_matrix(&pauli_y);
    println!("  Determinant: {}", pauli_y.determinant());

    let pauli_z = Matrix::new(2, 2, vec![
        Complex::new(1.0, 0.0),  Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),  Complex::new(-1.0, 0.0),
    ]);
    println!("Pauli Z matrix:");
    print_matrix(&pauli_z);
    println!("  Determinant: {}", pauli_z.determinant());

    // Hadamard gate (quantum computing)
    let sqrt2_inv = 1.0 / 2.0f64.sqrt();
    let hadamard = Matrix::new(2, 2, vec![
        Complex::new(sqrt2_inv, 0.0),   Complex::new(sqrt2_inv, 0.0),
        Complex::new(sqrt2_inv, 0.0),   Complex::new(-sqrt2_inv, 0.0),
    ]);
    println!("Hadamard gate:");
    print_matrix(&hadamard);
    println!("  Determinant: {}", hadamard.determinant());
    println!("  Is invertible? {}", hadamard.is_invertible());

    println!("\n=== All complex examples completed! ===");
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
