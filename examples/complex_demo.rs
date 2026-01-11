use num_complex::Complex;
use spectralize::Matrix;

fn main() {
    // Example 1: Create a complex matrix
    println!("=== Complex Number Matrix Demo ===\n");

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
    println!("Complex matrix:\n{:?}\n", m);

    // Example 2: Identity matrix with complex numbers
    let identity = Matrix::<Complex<f64>>::identity(3, 3);
    println!("Complex identity matrix:\n{:?}\n", identity);

    // Example 3: Zero matrix with complex numbers
    let zero = Matrix::<Complex<f64>>::zero(2, 3);
    println!("Complex zero matrix:\n{:?}\n", zero);

    // Example 4: Concatenation with complex matrices
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

    let concatenated = a.with_cols(&b);
    println!("Concatenated complex matrices:\n{:?}\n", concatenated);

    // Example 5: Different numeric types
    println!("=== Other Numeric Types ===\n");

    let f32_matrix = Matrix::new(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]);
    println!("f32 matrix:\n{:?}\n", f32_matrix);

    let i32_matrix = Matrix::new(2, 2, vec![1i32, 2, 3, 4]);
    println!("i32 matrix:\n{:?}\n", i32_matrix);

    println!("All matrix types are now fully generic!");
}
