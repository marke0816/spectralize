use spectralize::Matrix;

fn main() {
    println!("=== Matrix Transpose Demo ===\n");

    // Create a 2x3 matrix
    let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    println!("Original matrix (2x3):");
    for row in 0..m.rows() {
        print!("  ");
        for col in 0..m.cols() {
            print!("{:4.1} ", m.get(row, col));
        }
        println!();
    }

    // Transpose the matrix
    let m_t = m.transpose();
    println!("\nTransposed matrix (3x2):");
    for row in 0..m_t.rows() {
        print!("  ");
        for col in 0..m_t.cols() {
            print!("{:4.1} ", m_t.get(row, col));
        }
        println!();
    }

    println!("\n=== Matrix Trace Demo ===\n");

    // Create a 3x3 matrix
    let square = Matrix::new(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    println!("Square matrix (3x3):");
    for row in 0..square.rows() {
        print!("  ");
        for col in 0..square.cols() {
            print!("{:4.1} ", square.get(row, col));
        }
        println!();
    }

    let trace = square.trace();
    println!("\nTrace (sum of diagonal): {}", trace);
    println!("(1 + 5 + 9 = {})", trace);

    // Demonstrate with identity matrix
    println!("\n=== Identity Matrix Trace ===\n");
    let identity = Matrix::<f64>::identity(4, 4);
    println!("Identity matrix (4x4):");
    for row in 0..identity.rows() {
        print!("  ");
        for col in 0..identity.cols() {
            print!("{:4.1} ", identity.get(row, col));
        }
        println!();
    }

    let id_trace = identity.trace();
    println!("\nTrace of identity matrix: {}", id_trace);
    println!("(Always equals the dimension: {})", id_trace);
}
