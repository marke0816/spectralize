use spectralize::matrix::{ConstMatrix, Matrix};

fn main() {
    println!("=== ConstMatrix Demo ===\n");

    // Create matrices with compile-time dimensions
    println!("1. Creating matrices:");
    let a: ConstMatrix<f64, 2, 3> = ConstMatrix::new(vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ]);
    println!("Matrix A (2x3):");
    print_matrix(&a);

    let b: ConstMatrix<f64, 3, 2> = ConstMatrix::new(vec![
        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0,
    ]);
    println!("Matrix B (3x2):");
    print_matrix(&b);

    // Transpose - note how dimensions swap in the type
    println!("\n2. Transpose:");
    let at: ConstMatrix<f64, 3, 2> = a.transpose();
    println!("A^T (3x2):");
    print_matrix(&at);

    // Identity and zero matrices
    println!("\n3. Special matrices:");
    let identity: ConstMatrix<i32, 3, 3> = ConstMatrix::identity();
    println!("Identity (3x3):");
    print_matrix(&identity);

    let zero: ConstMatrix<i32, 2, 4> = ConstMatrix::zero();
    println!("Zero (2x4):");
    print_matrix(&zero);

    // Conversion to/from dynamic Matrix
    println!("\n4. Conversions:");
    let dyn_mat: Matrix<f64> = a.clone().into();
    println!("Converted to dynamic Matrix: {}x{}", dyn_mat.rows(), dyn_mat.cols());

    let const_mat: Result<ConstMatrix<f64, 2, 3>, _> = dyn_mat.try_into();
    println!("Converted back to ConstMatrix<2, 3>: {}", const_mat.is_ok());

    // This would be a compile-time error:
    // let wrong: ConstMatrix<f64, 2, 2> = ConstMatrix::identity();
    // let result = wrong + a; // Error: expected ConstMatrix<_, 2, 3>, found ConstMatrix<_, 2, 2>

    println!("\n5. Type safety:");
    println!("The type system enforces dimension compatibility!");
    println!("Operations like A + B require matching dimensions at compile time.");
    println!("Matrix multiplication A * B enforces inner dimension matches.");

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
