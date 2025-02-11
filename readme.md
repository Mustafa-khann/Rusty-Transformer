# Rust Transformer Implementation

A pure Rust implementation of the Transformer architecture as described in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). This implementation uses the ndarray crate for efficient tensor operations.

## Features

- Complete Transformer architecture implementation
- Modular components:
  - Multi-head Self-Attention
  - Positional Encoding
  - Layer Normalization
  - Feed-Forward Networks
  - Encoder and Decoder stacks
- Configurable hyperparameters
- Dropout support for regularization
- Masking support for both padding and look-ahead

## Architecture

The implementation follows the original Transformer architecture with the following components:

### Core Components
- **Positional Encoding**: Adds positional information to input embeddings
- **Multi-Head Attention**: Allows the model to jointly attend to information from different representation subspaces
- **Layer Normalization**: Normalizes the outputs of sub-layers
- **Feed-Forward Networks**: Processes the attention outputs through fully connected layers

### Main Modules
- **Encoder**: Processes the input sequence
- **Decoder**: Generates the output sequence
- **EncoderLayer**: Single layer of the encoder stack
- **DecoderLayer**: Single layer of the decoder stack

## Default Hyperparameters

```rust
const MAX_SEQ_LENGTH: usize = 512
const EMBEDDING_DIM: usize = 512
const NUM_HEADS: usize = 8
const FF_DIM: usize = 2048
const NUM_LAYERS: usize = 6
const DROPOUT_RATE: f32 = 0.1
```

## Usage

1. Add the following dependencies to your `Cargo.toml`:

```toml
[dependencies]
ndarray = "0.15"
rand = "0.8"
```

2. Create a new Transformer instance:

```rust
let transformer = Transformer::new(
    vocab_size,
    MAX_SEQ_LENGTH,
    EMBEDDING_DIM,
    NUM_LAYERS,
    NUM_HEADS,
    FF_DIM,
    DROPOUT_RATE
);
```

3. Forward pass through the model:

```rust
let output = transformer.forward(
    &input,
    &target,
    Some(&enc_padding_mask),
    Some(&look_ahead_mask),
    Some(&dec_padding_mask)
);
```

## Helper Functions

The implementation includes utility functions for creating masks:

- `create_padding_mask`: Creates mask for padding tokens
- `create_look_ahead_mask`: Creates mask to prevent attention to future tokens
- `softmax`: Implements the softmax function for attention scores

## Implementation Details

The implementation uses the following Rust-specific features:

- `ndarray` for efficient tensor operations
- Generic array types (`Array1`, `Array2`, `Array3`) for different dimensional tensors
- Result types for error handling
- Trait implementations for core functionality
- Memory-efficient operations using references

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## References

1. Vaswani, A., et al. (2017). ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
2. [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) 