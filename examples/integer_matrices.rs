use spectralize::Matrix;

fn main() {
    println!("=== Integer Matrix Examples (i32 and i64) ===\n");

    // ==================== Matrix Creation ====================
    println!("1. Matrix Creation");

    // i32 matrix from vector
    let a_i32 = Matrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]);
    println!("i32 matrix A (2x3):");
    print_matrix(&a_i32);

    // i64 identity matrix
    let identity_i64 = Matrix::<i64>::identity(3, 3);
    println!("i64 identity matrix (3x3):");
    print_matrix(&identity_i64);

    // Zero matrix
    let zero_i32 = Matrix::<i32>::zero(2, 2);
    println!("i32 zero matrix (2x2):");
    print_matrix(&zero_i32);

    // Permutation matrix
    let perm = Matrix::<i32>::perm(3, 3, vec![2, 3, 1]);
    println!("i32 permutation matrix (swaps columns):");
    print_matrix(&perm);

    // ==================== Element Access ====================
    println!("\n2. Element Access");

    let element = a_i32.get(0, 1);
    println!("Element at (0, 1): {}", element);

    // Checked access (returns Result)
    match a_i32.try_get(0, 1) {
        Ok(val) => println!("Safe access at (0, 1): {}", val),
        Err(e) => println!("Error: {:?}", e),
    }

    // Row access (zero-copy slice)
    let row = a_i32.row(0);
    println!("First row: {:?}", row);

    // Column iteration without allocation
    let col: Vec<i32> = a_i32.col_iter(1).copied().collect();
    println!("Second column: {:?}", col);

    // ==================== Matrix Modification ====================
    println!("\n3. Matrix Modification");

    let mut b_i32 = Matrix::<i32>::zero(2, 2);
    b_i32.set(0, 0, 10);
    b_i32.set(0, 1, 20);
    b_i32.set(1, 0, 30);
    b_i32.set(1, 1, 40);
    println!("Modified matrix B:");
    print_matrix(&b_i32);

    // ==================== Transpose ====================
    println!("\n4. Transpose");

    let transposed = a_i32.transpose();
    println!("Transpose of A (3x2):");
    print_matrix(&transposed);

    // ==================== Arithmetic Operations ====================
    println!("\n5. Arithmetic Operations");

    let c_i32 = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    let d_i32 = Matrix::new(2, 2, vec![5, 6, 7, 8]);

    println!("Matrix C:");
    print_matrix(&c_i32);
    println!("Matrix D:");
    print_matrix(&d_i32);

    // Addition
    let sum = &c_i32 + &d_i32;
    println!("C + D:");
    print_matrix(&sum);

    // Subtraction
    let diff = &c_i32 - &d_i32;
    println!("C - D:");
    print_matrix(&diff);

    // Matrix multiplication
    let product = &c_i32 * &d_i32;
    println!("C * D:");
    print_matrix(&product);

    // Scalar multiplication
    let scaled = &c_i32 * 2;
    println!("C * 2:");
    print_matrix(&scaled);

    // Scalar multiplication (scalar on left)
    let scaled_left = 3 * &c_i32;
    println!("3 * C:");
    print_matrix(&scaled_left);

    // ==================== Trace ====================
    println!("\n6. Trace (sum of diagonal)");

    let trace_c = c_i32.trace();
    println!("Trace of C: {}", trace_c);

    // ==================== Matrix Power ====================
    println!("\n7. Matrix Power");

    let c_squared = c_i32.pow(2);
    println!("C^2:");
    print_matrix(&c_squared);

    let c_cubed = c_i32.pow(3);
    println!("C^3:");
    print_matrix(&c_cubed);

    // ==================== Determinant (Exact Integer Arithmetic) ====================
    println!("\n8. Determinant (Bareiss Algorithm for Exact Integer Computation)");

    // Integer determinants use Bareiss algorithm for exact results (no floating-point)
    let det_c = c_i32.determinant();
    println!("Determinant of C: {}", det_c);

    // Example: 3x3 matrix
    let e_i32 = Matrix::new(3, 3, vec![1, 2, 3, 0, 1, 4, 5, 6, 0]);
    println!("Matrix E (3x3):");
    print_matrix(&e_i32);
    let det_e = e_i32.determinant();
    println!("Determinant of E: {}", det_e);

    // ==================== Invertibility ====================
    println!("\n9. Invertibility Check");

    println!("Is C invertible? {}", c_i32.is_invertible());
    println!("Is zero matrix invertible? {}", zero_i32.is_invertible());

    // Singular integer matrix
    let singular = Matrix::new(2, 2, vec![2, 4, 1, 2]);
    println!("Singular matrix:");
    print_matrix(&singular);
    println!("  Is invertible? {}", singular.is_invertible());
    println!("  Determinant: {}", singular.determinant());

    // ==================== Matrix Norms ====================
    println!("\n10. Matrix Norms");

    let f_i32 = Matrix::new(2, 3, vec![3, 4, 0, 0, 0, 5]);
    println!("Matrix F:");
    print_matrix(&f_i32);

    // Note: Frobenius norm requires Sqrt trait (only available for f32/f64)
    // For integer matrices, use 1-norm or infinity norm instead
    println!("1-norm (max column sum): {}", f_i32.norm_one());
    println!("Infinity norm (max row sum): {}", f_i32.norm_inf());

    // ==================== Cross Product (3D Integer Vectors) ====================
    println!("\n11. Cross Product (3D Integer Vectors)");

    // Integer vectors - exact arithmetic
    let i_vec = Matrix::new(3, 1, vec![1, 0, 0]); // i
    let j_vec = Matrix::new(3, 1, vec![0, 1, 0]); // j
    let k_vec = Matrix::new(3, 1, vec![0, 0, 1]); // k

    println!("Standard basis vectors:");
    println!("i = (1, 0, 0), j = (0, 1, 0), k = (0, 0, 1)");

    let i_cross_j = i_vec.cross(&j_vec).unwrap();
    println!("\ni × j:");
    print_matrix(&i_cross_j);
    println!("Expected: k = (0, 0, 1) ✓");

    let j_cross_k = j_vec.cross(&k_vec).unwrap();
    println!("\nj × k:");
    print_matrix(&j_cross_k);
    println!("Expected: i = (1, 0, 0) ✓");

    let k_cross_i = k_vec.cross(&i_vec).unwrap();
    println!("\nk × i:");
    print_matrix(&k_cross_i);
    println!("Expected: j = (0, 1, 0) ✓");

    // Practical example with integer coordinates
    let point1 = Matrix::new(3, 1, vec![2, 3, 4]);
    let point2 = Matrix::new(3, 1, vec![5, 6, 7]);
    let normal = point1.cross(&point2).unwrap();

    println!("\nVector p1:");
    print_matrix(&point1);
    println!("Vector p2:");
    print_matrix(&point2);
    println!("p1 × p2 (normal vector, exact):");
    print_matrix(&normal);
    println!(
        "Result: ({}, {}, {})",
        normal.get(0, 0),
        normal.get(1, 0),
        normal.get(2, 0)
    );

    // Anti-commutativity: a × b = -(b × a)
    let reverse = point2.cross(&point1).unwrap();
    println!("\np2 × p1:");
    print_matrix(&reverse);
    println!("Notice: p2 × p1 = -(p1 × p2) (anti-commutative)");

    // ==================== Concatenation ====================
    println!("\n12. Matrix Concatenation");

    let left = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    let right = Matrix::new(2, 1, vec![5, 6]);

    println!("Left matrix:");
    print_matrix(&left);
    println!("Right matrix:");
    print_matrix(&right);

    let h_concat = left.with_cols(&right);
    println!("Horizontal concatenation:");
    print_matrix(&h_concat);

    let top = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    let bottom = Matrix::new(1, 2, vec![5, 6]);

    let v_concat = top.with_rows(&bottom);
    println!("Vertical concatenation:");
    print_matrix(&v_concat);

    // Adding single row/column
    let g = Matrix::new(2, 2, vec![1, 2, 3, 4]);
    let new_row = vec![5, 6];
    let with_row = g.with_row_vec(&new_row);
    println!("Matrix with added row:");
    print_matrix(&with_row);

    let new_col = vec![7, 8];
    let with_col = g.with_col_vec(&new_col);
    println!("Matrix with added column:");
    print_matrix(&with_col);

    // ==================== Checked APIs ====================
    println!("\n13. Checked APIs (error handling)");

    let h = Matrix::new(2, 3, vec![1, 2, 3, 4, 5, 6]);

    // Safe get
    match h.try_get(0, 1) {
        Ok(val) => println!("try_get(0, 1): {}", val),
        Err(e) => println!("Error: {:?}", e),
    }

    match h.try_get(5, 5) {
        Ok(val) => println!("try_get(5, 5): {}", val),
        Err(e) => println!("Error: {:?}", e),
    }

    // Safe trace (requires square matrix)
    match h.try_trace() {
        Ok(val) => println!("Trace: {}", val),
        Err(e) => println!("Cannot compute trace: {:?}", e),
    }

    // Safe determinant
    match h.try_determinant() {
        Ok(val) => println!("Determinant: {}", val),
        Err(e) => println!("Cannot compute determinant: {:?}", e),
    }

    // Safe permutation (checks for duplicates and bounds)
    match Matrix::<i32>::try_perm(3, 3, vec![1, 2, 1]) {
        Ok(_) => println!("Permutation created"),
        Err(e) => println!("Invalid permutation: {:?}", e),
    }

    // ==================== i64 Examples ====================
    println!("\n14. i64 Examples (64-bit integers)");

    let a_i64 = Matrix::new(2, 2, vec![1i64, 2i64, 3i64, 4i64]);
    let b_i64 = Matrix::new(2, 2, vec![5i64, 6i64, 7i64, 8i64]);

    println!("i64 matrix A:");
    print_matrix(&a_i64);
    println!("i64 matrix B:");
    print_matrix(&b_i64);

    let sum_i64 = &a_i64 + &b_i64;
    println!("A + B (i64):");
    print_matrix(&sum_i64);

    println!("Determinant of A (i64): {}", a_i64.determinant());
    println!("Is A invertible (i64)? {}", a_i64.is_invertible());

    // Large integers (demonstrate i64 range)
    let large_i64 = Matrix::new(
        2,
        2,
        vec![1_000_000i64, 2_000_000i64, 3_000_000i64, 4_000_000i64],
    );
    println!("Large i64 matrix:");
    print_matrix(&large_i64);
    println!("Determinant: {}", large_i64.determinant());

    // ==================== Integer Overflow Warning ====================
    println!("\n15. Integer Overflow Considerations");
    println!("NOTE: Integer determinants use the Bareiss algorithm for exact computation.");
    println!("WARNING: Large matrices or large values may overflow. Consider using i64 or i128");
    println!("for better range, or use f64 for approximate results on very large computations.");

    // Safe example that won't overflow
    let safe = Matrix::new(3, 3, vec![1, 2, 3, 4, 5, 6, 7, 8, 10]);
    println!("\nSafe 3x3 matrix:");
    print_matrix(&safe);
    println!("Determinant: {}", safe.determinant());

    println!("\n=== All integer examples completed! ===");
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
