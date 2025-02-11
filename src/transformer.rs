use ndarray::{s, Array1, Array2, Array3, Array4, Axis};

use rand::prelude::*;
use rand_distr::Normal;


// Model hyperparameters
const MAX_SEQ_LENGTH: usize = 512;
const EMBEDDING_DIM: usize = 512;
const NUM_HEADS: usize = 8;
const FF_DIM: usize = 2048;
const NUM_LAYERS: usize = 6;
const DROPOUT_RATE: f32 = 0.1;


// Positional Encoding
pub struct PositionalEncoding {
    encoding: Array2<f32>,
}

impl PositionalEncoding {
    pub fn new(max_seq_length: usize, embedding_dim: usize) -> Self {
        let mut encoding = Array2::zeros((max_seq_length, embedding_dim));
        
        for pos in 0..max_seq_length {
            for i in 0..(embedding_dim/2) {
                let angle = pos as f32 / f32::powf(10000.0, (2 * i) as f32 / embedding_dim as f32);
                encoding[[pos, 2*i]] = angle.sin();
                encoding[[pos, 2*i+1]] = angle.cos();
            }
        }
        
        Self { encoding }
    }
    pub fn get_encoding(&self, seq_length: usize) -> Array2<f32> {
        self.encoding.slice(s![0..seq_length, ..]).to_owned()
    }
}

// Layer Normalization
pub struct LayerNorm {
    gamma: Array1<f32>,
    beta: Array1<f32>,
    eps: f32,
}

impl LayerNorm {
    pub fn new(size: usize) -> Self {
        Self {
            gamma: Array1::ones(size),
            beta: Array1::zeros(size),
            eps: 1e-5,
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        print_array_info("Input", x);
        print_array_info("Gamma", &self.gamma.clone().into_shape((1, self.gamma.len())).unwrap());
        
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1] / self.gamma.len();
        println!("batch_size: {}, seq_len: {}, gamma_len: {}", batch_size, seq_len, self.gamma.len());
        
        // Reshape input to [batch_size * seq_len, embedding_dim]
        let x_reshaped = match x.clone().into_shape((batch_size * seq_len, self.gamma.len())) {
            Ok(reshaped) => reshaped,
            Err(e) => {
                println!("Failed to reshape input: {:?}", e);
                println!("Input shape: {:?}, Target shape: {:?}", x.shape(), (batch_size * seq_len, self.gamma.len()));
                panic!("Reshape failed");
            }
        };
        
        // Calculate mean and variance over the last dimension
        let mean = match x_reshaped.mean_axis(Axis(1)) {
            Some(m) => m,
            None => {
                println!("Failed to calculate mean. Input shape: {:?}, Reshaped shape: {:?}", x.shape(), x_reshaped.shape());
                // Calculate mean manually if the ndarray method fails
                let sum = x_reshaped.sum_axis(Axis(1));
                sum / x_reshaped.shape()[1] as f32
            }
        };
        
        let var = x_reshaped.var_axis(Axis(1), 0.0);

        // Broadcast the mean and variance
        let mean_broadcast = mean.insert_axis(Axis(1));
        let var_broadcast = var.insert_axis(Axis(1));
        let gamma_broadcast = self.gamma.clone().insert_axis(Axis(0));
        let beta_broadcast = self.beta.clone().insert_axis(Axis(0));

        // Broadcast to match input dimensions
        let mean_broadcast = mean_broadcast.broadcast(x_reshaped.raw_dim()).unwrap();
        let var_broadcast = var_broadcast.broadcast(x_reshaped.raw_dim()).unwrap();
        let gamma_broadcast = gamma_broadcast.broadcast(x_reshaped.raw_dim()).unwrap();
        let beta_broadcast = beta_broadcast.broadcast(x_reshaped.raw_dim()).unwrap();

        // Normalize
        let var_eps = var_broadcast.mapv(|v| v + self.eps);
        let normalized = (x_reshaped - &mean_broadcast) / &var_eps.mapv(f32::sqrt);
        
        // Scale and shift
        let output = normalized * &gamma_broadcast + &beta_broadcast;
        
        // Clone output before reshaping for potential error reporting
        let output_shape = output.shape().to_vec();
        match output.into_shape(x.raw_dim()) {
            Ok(reshaped) => reshaped,
            Err(e) => {
                println!("Failed to reshape output: {:?}", e);
                println!("Output shape: {:?}, Target shape: {:?}", output_shape, x.raw_dim());
                panic!("Final reshape failed");
            }
        }
    }
}

// Add this helper function to print array info with more detail
fn print_array_info(name: &str, arr: &Array2<f32>) {
    println!("{} shape: {:?}", name, arr.shape());
    println!("{} first few values: {:?}", name, arr.slice(s![0..1, 0..5]));
    println!("{} strides: {:?}", name, arr.strides());
    println!("{} is standard layout: {}", name, arr.is_standard_layout());
    println!("{} total elements: {}", name, arr.len());
}

// Multi-Head Attention
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    w_query: Array2<f32>,
    w_key: Array2<f32>,
    w_value: Array2<f32>,
    w_output: Array2<f32>,
}

impl MultiHeadAttention {
    pub fn new(embedding_dim: usize, num_heads: usize) -> Self {
        let head_dim = embedding_dim / num_heads;
        
        let dist = Normal::new(0.0, 0.02).unwrap();
        let w_query = random_array2((embedding_dim, embedding_dim), dist);
        let w_key = random_array2((embedding_dim, embedding_dim), dist);
        let w_value = random_array2((embedding_dim, embedding_dim), dist);
        let w_output = random_array2((embedding_dim, embedding_dim), dist);
        
        Self {
            num_heads,
            head_dim,
            w_query,
            w_key,
            w_value,
            w_output,
        }
    }

    fn split_heads(&self, x: &Array2<f32>) -> Array4<f32> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        x.to_owned()
            .into_shape((batch_size, seq_len, self.num_heads, self.head_dim))
            .unwrap()
    }

    fn scaled_dot_product_attention(&self, query: &Array4<f32>, key: &Array4<f32>, value: &Array4<f32>, mask: Option<&Array2<f32>>) -> Array4<f32> {
        let scale = (self.head_dim as f32).powf(-0.5);
        
        // Reshape for matrix multiplication
        let batch_size = query.shape()[0];
        let num_heads = query.shape()[2];
        let seq_len_q = query.shape()[1];
        let seq_len_k = key.shape()[1];
        
        // Reshape to 3D tensors for matrix multiplication
        let q = query.to_owned()
            .into_shape((batch_size * num_heads, seq_len_q, self.head_dim))
            .unwrap();
        let k = key.to_owned()
            .into_shape((batch_size * num_heads, seq_len_k, self.head_dim))
            .unwrap();
        let v = value.to_owned()
            .into_shape((batch_size * num_heads, seq_len_k, self.head_dim))
            .unwrap();
        
        // Compute attention scores using matrix multiplication between q and k_t
        let mut scores = Array2::zeros((batch_size * num_heads * seq_len_q, seq_len_k));
        for i in 0..(batch_size * num_heads) {
            let q_slice = q.slice(s![i, .., ..]);
            let k_slice = k.slice(s![i, .., ..]);
            let score_slice = q_slice.dot(&k_slice.t());
            scores.slice_mut(s![i * seq_len_q..(i + 1) * seq_len_q, ..])
                .assign(&score_slice);
        }
        let mut scores = scores.into_shape((batch_size * num_heads, seq_len_q, seq_len_k)).unwrap() * scale;
        
        if let Some(mask) = mask {
            // Reshape mask to match attention scores dimensions
            let expanded_mask = mask.broadcast((batch_size * num_heads, seq_len_q, seq_len_k))
                .unwrap();
            scores = scores + expanded_mask;
        }

        let attention_weights = softmax(&scores.into_shape((batch_size * num_heads * seq_len_q, seq_len_k)).unwrap(), Axis(1))
            .into_shape((batch_size * num_heads, seq_len_q, seq_len_k))
            .unwrap();

        // Compute output using matrix multiplication between attention_weights and v
        let mut output = Array3::zeros((batch_size * num_heads, seq_len_q, self.head_dim));
        for i in 0..(batch_size * num_heads) {
            let weights_slice = attention_weights.slice(s![i, .., ..]);
            let v_slice = v.slice(s![i, .., ..]);
            let output_slice = weights_slice.dot(&v_slice);
            output.slice_mut(s![i, .., ..])
                .assign(&output_slice);
        }
        
        // Reshape back to 4D
        output.into_shape((batch_size, seq_len_q, num_heads, self.head_dim)).unwrap()
    }

    pub fn forward(&self, query: &Array2<f32>, key: &Array2<f32>, value: &Array2<f32>, mask: Option<&Array2<f32>>) -> Array2<f32> {
        let batch_size = query.shape()[0];
        let seq_len = query.shape()[1] / self.w_query.shape()[0];  // Adjust for flattened input
        
        // Reshape inputs to match weight matrices
        let query = query.clone().into_shape((batch_size * seq_len, self.w_query.shape()[0])).unwrap();
        let key = key.clone().into_shape((batch_size * seq_len, self.w_key.shape()[0])).unwrap();
        let value = value.clone().into_shape((batch_size * seq_len, self.w_value.shape()[0])).unwrap();
        
        // Linear projections
        let q = query.dot(&self.w_query);
        let k = key.dot(&self.w_key);
        let v = value.dot(&self.w_value);

        // Reshape back to include sequence length
        let q = q.into_shape((batch_size, seq_len, self.num_heads, self.head_dim)).unwrap();
        let k = k.into_shape((batch_size, seq_len, self.num_heads, self.head_dim)).unwrap();
        let v = v.into_shape((batch_size, seq_len, self.num_heads, self.head_dim)).unwrap();

        // Apply scaled dot-product attention
        let attention_output = self.scaled_dot_product_attention(&q, &k, &v, mask);

        // Reshape and project back
        let attention_output = attention_output
            .into_shape((batch_size, seq_len * self.num_heads * self.head_dim))
            .unwrap();

        // Final linear projection
        attention_output.dot(&self.w_output)
    }
}

// Feed Forward Network
pub struct FeedForward {
    w1: Array2<f32>,
    w2: Array2<f32>,
    b1: Array1<f32>,
    b2: Array1<f32>,
}

impl FeedForward {
    pub fn new(embedding_dim: usize, ff_dim: usize) -> Self {
        Self {
            w1: random_array2((embedding_dim, ff_dim), Normal::new(0.0, 0.02).unwrap()),
            w2: random_array2((ff_dim, embedding_dim), Normal::new(0.0, 0.02).unwrap()),
            b1: Array1::zeros(ff_dim),
            b2: Array1::zeros(embedding_dim),
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // First linear transformation followed by ReLU
        let hidden = x.dot(&self.w1) + &self.b1;
        let hidden = hidden.mapv(|x| x.max(0.0)); // ReLU activation

        // Second linear transformation
        hidden.dot(&self.w2) + &self.b2
    }
}

// Encoder Layer
pub struct EncoderLayer {
    self_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl EncoderLayer {
    pub fn new(embedding_dim: usize, num_heads: usize, ff_dim: usize) -> Self {
        Self {
            self_attention: MultiHeadAttention::new(embedding_dim, num_heads),
            feed_forward: FeedForward::new(embedding_dim, ff_dim),
            norm1: LayerNorm::new(embedding_dim),
            norm2: LayerNorm::new(embedding_dim),
        }
    }

    pub fn forward(&self, x: &Array2<f32>, mask: Option<&Array2<f32>>) -> Array2<f32> {
        // Self attention block
        let norm_x = self.norm1.forward(x);
        let attention_output = self.self_attention.forward(&norm_x, &norm_x, &norm_x, mask);
        let x = x + &attention_output; // Residual connection

        // Feed forward block
        let norm_x = self.norm2.forward(&x);
        let ff_output = self.feed_forward.forward(&norm_x);
        x + &ff_output // Residual connection
    }
}

// Decoder Layer
pub struct DecoderLayer {
    self_attention: MultiHeadAttention,
    cross_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
}

impl DecoderLayer {
    pub fn new(embedding_dim: usize, num_heads: usize, ff_dim: usize) -> Self {
        Self {
            self_attention: MultiHeadAttention::new(embedding_dim, num_heads),
            cross_attention: MultiHeadAttention::new(embedding_dim, num_heads),
            feed_forward: FeedForward::new(embedding_dim, ff_dim),
            norm1: LayerNorm::new(embedding_dim),
            norm2: LayerNorm::new(embedding_dim),
            norm3: LayerNorm::new(embedding_dim),
        }
    }

    pub fn forward(&self, x: &Array2<f32>, enc_output: &Array2<f32>,
        look_ahead_mask: Option<&Array2<f32>>, padding_mask: Option<&Array2<f32>>) -> Array2<f32> {
        // Self Attention Block
        let norm_x = self.norm1.forward(x);
        let self_attn = self.self_attention.forward(&norm_x, &norm_x, &norm_x, look_ahead_mask);
        let x = x + &self_attn;

        // Cross attention block
        let norm_x = self.norm2.forward(&x);
        let cross_attn = self.cross_attention.forward(&norm_x, enc_output, enc_output, padding_mask);
        let x = x + &cross_attn;

        // Feed forward block
        let norm_x = self.norm3.forward(&x);
        let ff_output = self.feed_forward.forward(&norm_x);
        x + &ff_output
    }
}

// Encoder 
pub struct Encoder {
    embedding: Array2<f32>,
    positional_encoding: PositionalEncoding,
    layers: Vec<EncoderLayer>,
    norm: LayerNorm,
    dropout_rate: f32,
}

impl Encoder {
    pub fn new(vocab_size: usize, max_seq_length: usize, embedding_dim: usize, num_layers: usize, num_heads: usize, ff_dim: usize, dropout_rate: f32) -> Self {
        let layers = (0..num_layers)
            .map(|_| EncoderLayer::new(embedding_dim, num_heads, ff_dim))
            .collect();

        Self {
            embedding: Array2::zeros((vocab_size, embedding_dim)),
            positional_encoding: PositionalEncoding::new(max_seq_length, embedding_dim),
            layers,
            norm: LayerNorm::new(embedding_dim),
            dropout_rate,
        }   
    }

    pub fn forward(&self, x: &Array2<f32>, mask: Option<&Array2<f32>>, training: bool) -> Array2<f32> {
        let batch_size = x.shape()[0];
        let seq_length = x.shape()[1];
        let embedding_dim = self.embedding.shape()[1];
        
        // Initialize output array for embeddings
        let mut embedded = Array2::zeros((batch_size * seq_length, embedding_dim));
        
        // Perform embedding lookup
        for i in 0..batch_size {
            for j in 0..seq_length {
                let idx = x[[i, j]] as usize;
                let row_idx = i * seq_length + j;
                embedded.row_mut(row_idx).assign(&self.embedding.row(idx));
            }
        }
        
        // Reshape to 3D (batch_size, seq_length, embedding_dim)
        let mut x = embedded.into_shape((batch_size, seq_length, embedding_dim)).unwrap();
        
        // Add positional encoding
        x = x + &self.positional_encoding.get_encoding(seq_length)
            .into_shape((1, seq_length, embedding_dim)).unwrap();

        // Apply dropout in 3D
        x = self.apply_dropout3d(&x, training);

        // Process through layers
        for layer in &self.layers {
            let x_2d = x.clone().into_shape((batch_size, seq_length * embedding_dim)).unwrap();
            let layer_output = layer.forward(&x_2d, mask);
            x = layer_output.into_shape((batch_size, seq_length, embedding_dim)).unwrap();
            x = self.apply_dropout3d(&x, training);
        }

        // Final normalization and reshape to 2D
        x.into_shape((batch_size, seq_length * embedding_dim)).unwrap()
    }

    pub fn apply_dropout3d(&self, x: &Array3<f32>, training: bool) -> Array3<f32> {
        if !training || self.dropout_rate == 0.0 {
            return x.to_owned();
        }
        
        let mut rng = thread_rng();
        let mask = Array3::from_shape_fn(x.raw_dim(), |_| {
            if rng.gen::<f32>() > self.dropout_rate { 1.0 } else { 0.0 }
        });
        
        x * &mask * (1.0 / (1.0 - self.dropout_rate))
    }
}

// Decoder
pub struct Decoder {
    embedding: Array2<f32>,
    positional_encoding: PositionalEncoding,
    layers: Vec<DecoderLayer>,
    norm: LayerNorm,
    dropout_rate: f32,
}

impl Decoder {
    pub fn new(vocab_size: usize, max_seq_length: usize, embedding_dim: usize, num_layers: usize, num_heads: usize, ff_dim: usize, dropout_rate: f32) -> Self {
        let layers = (0..num_layers)
            .map(|_| DecoderLayer::new(embedding_dim, num_heads, ff_dim))
            .collect();

        Self {
            embedding: Array2::zeros((vocab_size, embedding_dim)),
            positional_encoding: PositionalEncoding::new(max_seq_length, embedding_dim),
            layers, 
            norm: LayerNorm::new(embedding_dim),
            dropout_rate,
        }
    }

    pub fn forward(&self, x: &Array2<f32>, enc_output: &Array2<f32>,
        look_ahead_mask: Option<&Array2<f32>>, padding_mask: Option<&Array2<f32>>, training: bool) -> Array2<f32> {
        let batch_size = x.shape()[0];
        let seq_length = x.shape()[1];
        let embedding_dim = self.embedding.shape()[1];
        
        // Initialize output array for embeddings
        let mut embedded = Array2::zeros((batch_size * seq_length, embedding_dim));
        
        // Perform embedding lookup
        for i in 0..batch_size {
            for j in 0..seq_length {
                let idx = x[[i, j]] as usize;
                let row_idx = i * seq_length + j;
                embedded.row_mut(row_idx).assign(&self.embedding.row(idx));
            }
        }
        
        // Reshape to 3D (batch_size, seq_length, embedding_dim)
        let mut x = embedded.into_shape((batch_size, seq_length, embedding_dim)).unwrap();
        
        // Add positional encoding
        x = x + &self.positional_encoding.get_encoding(seq_length)
            .into_shape((1, seq_length, embedding_dim)).unwrap();

        // Apply dropout in 3D
        x = self.apply_dropout3d(&x, training);

        // Process through layers
        for layer in &self.layers {
            let x_2d = x.clone().into_shape((batch_size, seq_length * embedding_dim)).unwrap();
            let layer_output = layer.forward(&x_2d, enc_output, look_ahead_mask, padding_mask);
            x = layer_output.into_shape((batch_size, seq_length, embedding_dim)).unwrap();
            x = self.apply_dropout3d(&x, training);
        }

        // Final normalization and reshape to 2D
        x.into_shape((batch_size, seq_length * embedding_dim)).unwrap()
    }

    pub fn apply_dropout3d(&self, x: &Array3<f32>, training: bool) -> Array3<f32> {
        if !training || self.dropout_rate == 0.0 {
            return x.to_owned();
        }
        
        let mut rng = thread_rng();
        let mask = Array3::from_shape_fn(x.raw_dim(), |_| {
            if rng.gen::<f32>() > self.dropout_rate { 1.0 } else { 0.0 }
        });
        
        x * &mask * (1.0 / (1.0 - self.dropout_rate))
    }
}

// Transformer
pub struct Transformer {
    encoder: Encoder, 
    decoder: Decoder, 
    final_layer: Array2<f32>,
    vocab_size: usize,
}

impl Transformer {
    pub fn new(vocab_size: usize, max_seq_length: usize, embedding_dim: usize, num_layers: usize, num_heads: usize, ff_dim: usize, dropout_rate: f32) -> Self {
        // Initialize embeddings with random values
        let embedding_dist = Normal::new(0.0, 0.02).unwrap();
        let shared_embedding = random_array2((vocab_size, embedding_dim), embedding_dist);
        
        let mut encoder = Encoder::new(vocab_size, max_seq_length, embedding_dim, num_layers, num_heads, ff_dim, dropout_rate);
        let mut decoder = Decoder::new(vocab_size, max_seq_length, embedding_dim, num_layers, num_heads, ff_dim, dropout_rate);
        
        // Set the embeddings
        encoder.embedding = shared_embedding.clone();
        decoder.embedding = shared_embedding.clone();
        
        Self {
            encoder,
            decoder,
            final_layer: shared_embedding, // Use the same embedding for the output layer
            vocab_size,
        }
    }

    pub fn forward(&self, input: &Array2<f32>, target: &Array2<f32>,
        enc_padding_mask: Option<&Array2<f32>>,
        look_ahead_mask: Option<&Array2<f32>>,
        dec_padding_mask: Option<&Array2<f32>>,
        training: bool) -> Array2<f32> {
        
        let enc_output = self.encoder.forward(input, enc_padding_mask, training);
        let dec_output = self.decoder.forward(target, &enc_output, 
            look_ahead_mask, dec_padding_mask, training);

        let batch_size = dec_output.shape()[0];
        let seq_len = dec_output.shape()[1];
        
        // Compute logits and reshape to 3D for proper softmax
        let logits = dec_output.dot(&self.final_layer)
            .into_shape((batch_size, seq_len, self.vocab_size))
            .unwrap();
        
        // Apply 3D softmax and reshape back to 2D if needed
        softmax3d(&logits, Axis(2))
            .into_shape((batch_size, seq_len * self.vocab_size))
            .unwrap()
    }
}

pub fn create_padding_mask(seq: &Array2<f32>) -> Array2<f32> {
    // Create a 2D mask where padded positions (0.0) become a large negative value
    seq.map(|&x| if x == 0.0 { f32::NEG_INFINITY } else { 0.0 })
}

pub fn create_look_ahead_mask(size:usize) -> Array2<f32> {
    let mut mask = Array2::ones((size, size));
    for i in 0..size {
        for j in 0..size {
                if j > i {
                mask[[i, j]] = 0.0;
            }
        }
    }
    mask
}

pub fn softmax(x: &Array2<f32>, axis: Axis) -> Array2<f32> {
    let max = x.fold_axis(axis, std::f32::NEG_INFINITY, |&acc, &x| acc.max(x));
    let exp = x - &max.insert_axis(axis);
    let exp = exp.mapv(f32::exp);
    let sum = exp.sum_axis(axis);
    exp / &sum.insert_axis(axis)
}

// Add a new softmax function for 3D arrays
pub fn softmax3d(x: &Array3<f32>, axis: Axis) -> Array3<f32> {
    let max = x.fold_axis(axis, std::f32::NEG_INFINITY, |&acc, &x| acc.max(x));
    let exp = x - &max.insert_axis(axis);
    let exp = exp.mapv(f32::exp);
    let sum = exp.sum_axis(axis);
    exp / &sum.insert_axis(axis)
}

fn random_array2(shape: (usize, usize), dist: Normal<f32>) -> Array2<f32> {
    let mut rng = thread_rng();
    Array2::from_shape_fn(shape, |_| dist.sample(&mut rng))
}