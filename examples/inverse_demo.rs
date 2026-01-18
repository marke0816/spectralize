use spectralize::matrix::Matrix;

fn main() {
    // Example 1: Basic 2x2 inverse
    println!("=== Example 1: Basic 2x2 matrix inverse ===");
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    println!("A = \n{:?}", a);

    let a_inv = a.inverse();
    println!("A^(-1) = \n{:?}", a_inv);

    let identity = &a * &a_inv;
    println!("A * A^(-1) = \n{:?}", identity);
    let expected = Matrix::identity(2, 2);
    println!("A * A^(-1) ≈ I: {}", identity.approx_eq(&expected, 1e-10));

    // Example 2: 3x3 matrix with custom tolerance
    println!("\n=== Example 2: 3x3 matrix inverse ===");
    let b = Matrix::new(3, 3, vec![
        1.0, 2.0, 3.0,
        0.0, 1.0, 4.0,
        5.0, 6.0, 0.0,
    ]);
    println!("B = \n{:?}", b);

    let b_inv = b.inverse();
    println!("B^(-1) = \n{:?}", b_inv);

    // Example 3: Singular matrix handling
    println!("\n=== Example 3: Singular matrix (should fail) ===");
    let singular = Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
    println!("Singular matrix = \n{:?}", singular);

    match singular.try_inverse() {
        Ok(_) => println!("Unexpectedly got inverse!"),
        Err(e) => println!("Correctly detected as singular: {:?}", e),
    }

    // Example 4: Nearly singular matrix with tolerance control
    println!("\n=== Example 4: Nearly singular matrix with tolerance ===");
    let nearly_singular = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 1.0 + 1e-10]);
    println!("Nearly singular matrix (det ≈ 1e-10)");

    // Strict tolerance: accepts as invertible
    match nearly_singular.try_inverse_with_tol(1e-12) {
        Ok(_) => println!("With strict tolerance (1e-12): invertible ✓"),
        Err(_) => println!("With strict tolerance (1e-12): singular"),
    }

    // Loose tolerance: rejects as singular
    match nearly_singular.try_inverse_with_tol(1e-8) {
        Ok(_) => println!("With loose tolerance (1e-8): invertible"),
        Err(_) => println!("With loose tolerance (1e-8): singular ✓"),
    }

    // Example 5: Larger matrix (5x5)
    println!("\n=== Example 5: 5x5 lower triangular matrix ===");
    let large = Matrix::new(5, 5, vec![
        1.0, 0.0, 0.0, 0.0, 0.0,
        2.0, 1.0, 0.0, 0.0, 0.0,
        3.0, 2.0, 1.0, 0.0, 0.0,
        4.0, 3.0, 2.0, 1.0, 0.0,
        5.0, 4.0, 3.0, 2.0, 1.0,
    ]);

    let large_inv = large.inverse();
    let verify = &large * &large_inv;
    let identity_5 = Matrix::identity(5, 5);
    println!("5x5 matrix correctly inverted: {}", verify.approx_eq(&identity_5, 1e-10));

    println!("\n=== All examples completed successfully! ===");
}
