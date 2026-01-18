use spectralize::matrix::ConstMatrix;

fn main() {
    println!("=== ConstMatrix Arithmetic Demo ===\n");

    // Addition with compile-time dimension checking
    println!("1. Addition (dimensions enforced at compile time):");
    let a: ConstMatrix<i32, 2, 3> = ConstMatrix::new(vec![
        1, 2, 3,
        4, 5, 6,
    ]);
    let b: ConstMatrix<i32, 2, 3> = ConstMatrix::new(vec![
        10, 20, 30,
        40, 50, 60,
    ]);
    println!("Matrix A:");
    print_matrix(&a);
    println!("Matrix B:");
    print_matrix(&b);

    let c = &a + &b;
    println!("A + B:");
    print_matrix(&c);

    // This would be a compile error:
    // let wrong: ConstMatrix<i32, 3, 2> = ConstMatrix::zero();
    // let result = a + wrong; // Error: mismatched types

    // Subtraction
    println!("\n2. Subtraction:");
    let d = &b - &a;
    println!("B - A:");
    print_matrix(&d);

    // Matrix multiplication with compile-time inner dimension checking
    println!("\n3. Matrix Multiplication (inner dimensions enforced):");
    let m1: ConstMatrix<i32, 2, 3> = ConstMatrix::new(vec![
        1, 2, 3,
        4, 5, 6,
    ]);
    let m2: ConstMatrix<i32, 3, 2> = ConstMatrix::new(vec![
        7, 8,
        9, 10,
        11, 12,
    ]);
    println!("Matrix M1 (2x3):");
    print_matrix(&m1);
    println!("Matrix M2 (3x2):");
    print_matrix(&m2);

    let m3: ConstMatrix<i32, 2, 2> = m1 * m2;
    println!("M1 * M2 (result is 2x2):");
    print_matrix(&m3);

    // This would be a compile error:
    // let wrong: ConstMatrix<i32, 2, 2> = ConstMatrix::zero();
    // let result = m1 * wrong; // Error: expected ConstMatrix<_, 3, _>, found ConstMatrix<_, 2, _>

    // Scalar multiplication
    println!("\n4. Scalar Multiplication:");
    let s1: ConstMatrix<f64, 2, 2> = ConstMatrix::new(vec![
        1.0, 2.0,
        3.0, 4.0,
    ]);
    println!("Matrix S1:");
    print_matrix(&s1);

    let s2 = &s1 * 2.5;
    println!("S1 * 2.5:");
    print_matrix(&s2);

    let s3 = 0.5 * &s1;
    println!("0.5 * S1:");
    print_matrix(&s3);

    // Combining operations
    println!("\n5. Combined Operations:");
    let identity: ConstMatrix<f64, 2, 2> = ConstMatrix::identity();
    println!("Identity (2x2):");
    print_matrix(&identity);

    // (I + I) * 2.0 = 4*I
    let result = (&identity + &identity) * 2.0;
    println!("(I + I) * 2.0:");
    print_matrix(&result);

    // Working with different element types
    println!("\n6. Complex Numbers:");
    use num_complex::Complex;
    let c1: ConstMatrix<Complex<f64>, 2, 2> = ConstMatrix::new(vec![
        Complex::new(1.0, 2.0),
        Complex::new(3.0, 4.0),
        Complex::new(5.0, 6.0),
        Complex::new(7.0, 8.0),
    ]);
    let c2: ConstMatrix<Complex<f64>, 2, 2> = ConstMatrix::new(vec![
        Complex::new(1.0, -1.0),
        Complex::new(2.0, -2.0),
        Complex::new(3.0, -3.0),
        Complex::new(4.0, -4.0),
    ]);
    println!("Complex Matrix C1:");
    print_complex_matrix(&c1);
    println!("Complex Matrix C2:");
    print_complex_matrix(&c2);

    let c3 = c1 + c2;
    println!("C1 + C2:");
    print_complex_matrix(&c3);

    println!("\n=== Key Benefits ===");
    println!("✓ Dimension mismatches caught at COMPILE TIME");
    println!("✓ No runtime overhead for dimension checks");
    println!("✓ Type-safe matrix operations");
    println!("✓ Works with all MatrixElement types (int, float, complex)");

    println!("\n=== Demo Complete ===");
}

fn print_matrix<T: std::fmt::Display, const R: usize, const C: usize>(m: &ConstMatrix<T, R, C>)
where
    T: spectralize::matrix::MatrixElement + std::fmt::Debug,
{
    for row in 0..R {
        print!("  [");
        for col in 0..C {
            print!("{:6}", format!("{}", m.get_ref(row, col)));
            if col < C - 1 {
                print!(" ");
            }
        }
        println!("]");
    }
}

fn print_complex_matrix<const R: usize, const C: usize>(
    m: &ConstMatrix<num_complex::Complex<f64>, R, C>,
) {
    for row in 0..R {
        print!("  [");
        for col in 0..C {
            let val = m.get_ref(row, col);
            if val.im >= 0.0 {
                print!("{:>8}", format!("{}+{}i", val.re, val.im));
            } else {
                print!("{:>8}", format!("{}{}i", val.re, val.im));
            }
            if col < C - 1 {
                print!(" ");
            }
        }
        println!("]");
    }
}
