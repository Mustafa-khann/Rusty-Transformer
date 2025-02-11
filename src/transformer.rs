use std::ops::Mul;
use ndarray::{Array, Array1, Array2, Array3, Axis};
use std::f32::consts::PI;

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
        let mut encoding = Array2::zeros((max_seq_length, d_model));

        for pos in 0..max_seq_length {
            for i in 0..d_model {
                let angle = pos as f32 / f32::powf(10000.0, (2 * i) as f32 / d_model as f32);
                encoding[[pos, 2*i]] = angle.sin();
                encoding[[pos, 2*i+1]] = angle.cos(); 
            }
        }

        Self {encoding}
    }
    pub fn get_encoding(&self, seq_length: usize) -> Array2<f32> {
        self.encoding.slice(s![0..seq_length, ..]).to_owned()
    }
}

// Layer Normalization
pub struct LayerNorm {
    gamma: Array2<f32>,
    beta: Array2<f32>,
    eps: f32,
}

impl LayerNorm {
    pub fn new(size: usize) -> Self {
        Self {
            gamma: Array2::ones(size),
            beta: Array2::zeros(size),
            eps: 1e-5,
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mean = x.mean_axis(Axis(1)).unwrap();
        let var = x.var_axis(Axis(1), 0.0);

        // Broadcast the mean and variance
        let mean_broadcast =&mean.broadcast((x.shape()[0], x.shape()[1])).unwrap();
        let var_broadcast = &var.broadcast((x.shape()[0], x.shape()[1])).unwrap();

        let normalized = (x - mean_broadcast) / (var_broadcast + self.eps).mapv(f32::sqrt);
        &normalized * &self.gamma + &self.beta
    }
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

        // Initialize weights with random values
        let w_query = Array2::zeros((embedding_dim, embedding_dim));
        let w_key = Array2::zeros((embedding_dim, embedding_dim));
        let w_value = Array2::zeros((embedding_dim, embedding_dim));
        let w_output = Array2::zeros((embedding_dim, embedding_dim));

        Self {
            num_heads,
            head_dim, 
            w_query,
            w_key,
            w_value,
            w_output,
        }
    }

    fn split_heads(&self, x: &Array2<f32>) -> Array3<f32> {
        let mut x = x.into_shape((batch_size, -1, self.num_heads, self.head_dim)).unwrap();
        x.permute_axes([0,2,1,3])
    }

    fn scaled_dot_product_attention(&self, query: &Array3<f32>, key: &Array3<f32>, value: &Array3<f32>, mask: Option<&Array2<f32>>) -> Array3<f32> {
        let scale = (self.head_dim as f32).powf(-0.5);
        let mut scores = query.dot(key.t()) * scale;

        if let Some(mask) = mask {
            // Apply mask to the scores
            scores = scores + mask;
        }

        let attention_weights = softmax(&scores, Axis(2));
        attention_weights.dot(value);
    } 

    pub fn forward(&self, query: &Array2<f32>, key: &Array2<f32>, value: &Array2<f32>, mask: Option<&Array2<f32>>) -> Array2<f32> {
        let batch_size = query.shape()[0];
        
        // Linear projections
        let q = query.dot(&self.w_query);
        let k = key.dot(&self.w_key);
        let v = value.dot(&self.w_value);

        // Split heads
        let q = self.split_heads(&q);
        let k = self.split_heads(&k);
        let v = self.split_heads(&v);

        // Apply scaled dot-product attention
        let attention_output = self.scaled_dot_product_attention(&q, &k, &v, mask);

        // Reshape back to original dimensions
        let mut attention_output = attention_output.permute_axes([0, 2, 1, 3]);
        let attention_output = attention_output.into_shape((batch_size, -1, self.num_heads * self.head_dim)).unwrap();

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
            w1: Array2::zeros((embedding_dim, ff_dim)),
            w2: Array2::zeros((ff_dim, embedding_dim)),
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
        let x = x + *self_attn;

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

    pub fn forward(&self, x:&Array2<f32>, mask: Option<&Array2<f32>>) -> Array2<f32> {
        let mut x = x.dot(&self.embedding);
        let seq_length = x.shape()[1];
        x = x + self.positional_encoding.get_encoding(seq_length);

        for layer in &self.layers {
            x = layer.forward(&x, mask);
        }

        self.norm.forward(&x);
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
        look_ahead_mask: Option<&Array2<f32>>, padding_mask: Option<&Array2<f32>>) -> Array2<f32> {
        let mut x = x.dot(&self.embedding);
        let seq_length = x.shape()[1];
        x = x + self.positional_encoding.get_encoding(seq_length);

        for layer in &self.layers {
        x = layer.forward(&x, enc_output, look_ahead_mask, padding_mask);
        }

        self.norm.forward(&x)
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
        Self {
            encoder: Encoder::new(vocab_size, max_seq_length, embedding_dim, num_layers, num_heads, ff_dim, dropout_rate),
            decoder: Decoder::new(vocab_size, max_seq_length, embedding_dim, num_layers, num_heads, ff_dim, dropout_rate),
            final_layer: Array2::zeros((embedding_dim, vocab_size)),
            vocab_size,
        }
    }

    pub fn forward(&self, input: &Array2<f32>, target: &Array2<f32>, enc_padding_mask: Option<&Array2<f32>>, look_ahead_mask: Option<&Array2<f32>>, dec_padding_mask: Option<&Array2<f32>>) -> Array2<f32> {
        // Encoder
        let enc_output = self.encoder.forward(input, enc_padding_mask);

        // Decoder 
        let dec_output = self.decoder.forward(target, &enc_output, look_ahead_mask, dec_padding_mask);

        // Final linear layer
        let logits = dec_output.dot(&self.final_layer);

        // Apply softmax
        softmax(&logits, Axis(2))
    }
}

pub fn create_padding_mask(seq: &Array2<f32>) -> Array2<f32> {
    let mask = seq.map(|&x| if x == 0.0 {1.0} else {0.0});
    mask.insert_axis(Axis(1))
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