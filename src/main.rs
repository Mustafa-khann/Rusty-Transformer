mod transformer;

use ndarray::{Array2, Axis};
use transformer::*;
use ndarray::{s};



fn main() {
    // Create a small transformer for testing
    let vocab_size = 1000;
    let max_seq_length = 50;
    let embedding_dim = 64;
    let num_layers = 2;
    let num_heads = 4;
    let ff_dim = 128;
    let dropout_rate = 0.1;

    let transformer = Transformer::new(
        vocab_size,
        max_seq_length,
        embedding_dim,
        num_layers,
        num_heads,
        ff_dim,
        dropout_rate,
    );

    // Create sample input sequences
    let batch_size = 2;
    let input_seq_len = 10;
    let target_seq_len = 8;

    // Create random input data (token indices should be integers)
    let input = Array2::from_shape_fn((batch_size, input_seq_len), |_| {
        (rand::random::<f32>() * (vocab_size - 1) as f32).floor() as i32 as f32
    });
    let target = Array2::from_shape_fn((batch_size, target_seq_len), |_| {
        (rand::random::<f32>() * (vocab_size - 1) as f32).floor() as i32 as f32
    });

    // Ensure values are in valid range
    println!("Sample input values:");
    println!("{:?}", input.slice(s![0, 0..5]));  // Print first 5 values of first sequence

    // Create masks
    let enc_padding_mask = Some(create_padding_mask(&input));
    let look_ahead_mask = Some(create_look_ahead_mask(target_seq_len));
    let dec_padding_mask = Some(create_padding_mask(&input));

    // Add debug prints before the forward pass
    println!("Input shape before forward: {:?}", input.shape());
    println!("First few input values: {:?}", input.slice(s![0, 0..5]));

    // Forward pass
    let output = transformer.forward(
        &input,
        &target,
        enc_padding_mask.as_ref(),
        look_ahead_mask.as_ref(),
        dec_padding_mask.as_ref(),
        true, // training mode
    );

    // Print output shape and check properties
    println!("Input shape: {:?}", input.shape());
    println!("Target shape: {:?}", target.shape());
    println!("Output shape: {:?}", output.shape());
    
    // Check if probabilities sum to 1 for each position
    let prob_sums = output.sum_axis(Axis(1));
    println!("\nProbability sums (should be close to 1.0):");
    println!("{:?}", prob_sums);

    // Print a sample of the output probabilities
    println!("\nSample output probabilities (first sequence, first 10 values):");
    println!("{:?}", output.slice(s![0, 0..10]));
}
