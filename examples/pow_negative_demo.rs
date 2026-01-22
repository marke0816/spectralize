use spectralize::matrix::Matrix;

fn main() {
    println!("=== Matrix Power with Negative Exponents Demo ===\n");

    // Example 1: Basic usage with diagonal matrix
    println!("Example 1: Diagonal matrix");
    let a = Matrix::new(2, 2, vec![2.0, 0.0, 0.0, 3.0]);
    println!("A = {:?}", a);

    let a_squared = a.pow(2);
    println!("A^2 = {:?}", a_squared);

    let a_inv = a.pow(-1);
    println!("A^(-1) = {:?}", a_inv);

    let a_inv_squared = a.pow(-2);
    println!("A^(-2) = {:?}\n", a_inv_squared);

    // Example 2: Verify A^2 * A^(-2) = I
    println!("Example 2: Verifying A^2 * A^(-2) = I");
    let product = &a_squared * &a_inv_squared;
    let identity = Matrix::identity(2, 2);
    println!("A^2 * A^(-2) = {:?}", product);
    println!("Is identity? {}\n", product.approx_eq(&identity, 1e-10));

    // Example 3: General 3x3 matrix
    println!("Example 3: General 3x3 matrix");
    let b = Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0]);
    println!("B = {:?}", b);

    let b_neg_one = b.pow(-1);
    println!("B^(-1) = {:?}", b_neg_one);

    let verify = &b * &b_neg_one;
    let identity_3 = Matrix::identity(3, 3);
    println!("B * B^(-1) = {:?}", verify);
    println!("Is identity? {}\n", verify.approx_eq(&identity_3, 1e-10));

    // Example 4: Using try_pow for error handling
    println!("Example 4: Error handling with try_pow");
    let singular = Matrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
    println!("Singular matrix = {:?}", singular);

    match singular.try_pow(-1) {
        Ok(_) => println!("Unexpectedly got inverse!"),
        Err(e) => println!("Correctly detected error: {:?}\n", e),
    }

    // Example 5: Different types
    println!("Example 5: f32 matrices");
    let c = Matrix::new(2, 2, vec![2.0f32, 0.0, 0.0, 3.0]);
    let c_neg_one = c.pow(-1);
    println!("C^(-1) = {:?}", c_neg_one);
    let expected = Matrix::new(2, 2, vec![0.5f32, 0.0, 0.0, 1.0 / 3.0]);
    println!("Correct? {}\n", c_neg_one.approx_eq(&expected, 1e-5));

    // Example 6: Higher negative powers
    println!("Example 6: Higher negative powers");
    let d = Matrix::new(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
    let d_neg_three = d.pow(-3);
    println!("D^(-3) = {:?}", d_neg_three);

    // Verify: D^3 * D^(-3) = I
    let d_cubed = d.pow(3);
    let product2 = &d_cubed * &d_neg_three;
    println!(
        "D^3 * D^(-3) ≈ I? {}\n",
        product2.approx_eq(&identity, 1e-10)
    );

    println!("=== All examples completed successfully! ===");
}
