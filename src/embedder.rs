use ort::session::Session;
use ort::value::Value;
use tokenizers::Tokenizer;
use ndarray::{Array2, Axis, ArrayViewD, IxDyn};
use std::sync::Mutex;

pub struct TextEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    pub dim: usize,
}

impl TextEmbedder {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Self {
        let session = Session::builder()
            .unwrap()
            .commit_from_file(model_path)
            .unwrap();

        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

        let mut embedder = Self { 
            session: Mutex::new(session), 
            tokenizer,
            dim: 0,
        };

        // Perform a dry run to auto-detect dimension
        let dummy_vec = embedder.embed("test");
        embedder.dim = dummy_vec.len();

        embedder
    }

    pub fn embed(&self, text: &str) -> Vec<f32> {
        let encoding = self.tokenizer.encode(text, true).unwrap();
        let input_ids = encoding.get_ids().iter().map(|&id| id as i64).collect::<Vec<_>>();
        let attention_mask = encoding.get_attention_mask().iter().map(|&mask| mask as i64).collect::<Vec<_>>();
        let token_type_ids = encoding.get_type_ids().iter().map(|&id| id as i64).collect::<Vec<_>>();

        let batch_size = 1;
        let seq_len = input_ids.len();

        let input_ids_array = Array2::from_shape_vec((batch_size, seq_len), input_ids).unwrap();
        let attention_mask_array = Array2::from_shape_vec((batch_size, seq_len), attention_mask).unwrap();
        let token_type_ids_array = Array2::from_shape_vec((batch_size, seq_len), token_type_ids).unwrap();

        let input_ids_val = Value::from_array(input_ids_array).unwrap();
        let attention_mask_val = Value::from_array(attention_mask_array).unwrap();
        let token_type_ids_val = Value::from_array(token_type_ids_array).unwrap();

        let mut session_lock = self.session.lock().expect("Failed to lock session");
        let outputs = session_lock.run(ort::inputs![
            "input_ids" => input_ids_val,
            "attention_mask" => attention_mask_val,
            "token_type_ids" => token_type_ids_val,
        ]).unwrap();

        let output_tensor = outputs["last_hidden_state"].try_extract_tensor::<f32>().unwrap();
        let (shape, data) = output_tensor;
        
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let view = ArrayViewD::from_shape(IxDyn(&dims), data).unwrap();
        
        let pooled = view.mean_axis(Axis(1)).unwrap();
        let mut vector: Vec<f32> = pooled.iter().cloned().collect();

        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for x in vector.iter_mut() {
                *x /= norm;
            }
        }
        
        vector
    }
}
