mod transformer;

use ndarray::{Array2, Axis};
use transformer::*;
use ndarray::{s, Array1, Array3, Array4};
use rand::prelude::*;
use rand_distr::Normal;

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

    // Create sample input sequences with padding
    let batch_size = 2;
    let input_seq_len = 10;
    let target_seq_len = 8;

    // Create input with actual padding (zeros)
    let mut input = Array2::zeros((batch_size, input_seq_len));
    // Fill first row with 1s up to position 7 (leaving 3 padding tokens)
    input.slice_mut(s![0, 0..7]).fill(1.0);
    // Fill second row with 1s up to position 5 (leaving 5 padding tokens)
    input.slice_mut(s![1, 0..5]).fill(1.0);

    // Create target with padding
    let mut target = Array2::zeros((batch_size, target_seq_len));
    // Fill first row with 1s up to position 6 (leaving 2 padding tokens)
    target.slice_mut(s![0, 0..6]).fill(1.0);
    // Fill second row with 1s up to position 4 (leaving 4 padding tokens)
    target.slice_mut(s![1, 0..4]).fill(1.0);

    // Create masks
    let enc_padding_mask = Some(create_padding_mask(&input));
    let look_ahead_mask = Some(create_look_ahead_mask(target_seq_len));
    let dec_padding_mask = Some(create_padding_mask(&input));

    // Forward pass
    let output = transformer.forward(
        &input,
        &target,
        enc_padding_mask.as_ref(),
        look_ahead_mask.as_ref(),
        dec_padding_mask.as_ref(),
        true,  // training mode
    );

    // Print output shape and check properties
    println!("Output shape: {:?}", output.shape());
    println!("Expected shape: [{}, {}, {}]", batch_size, target_seq_len, vocab_size);

    // Check if probabilities sum to 1 for each position
    let prob_sums = output.sum_axis(Axis(2));
    println!("Probability sums (should be close to 1.0):");
    println!("{:?}", prob_sums);
}
