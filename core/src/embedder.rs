use tokenizers::Tokenizer;
use ndarray::{Array2, Axis, ArrayViewD, IxDyn, s};
use ort::session::Session;
use ort::value::Value;
use ort::session::builder::GraphOptimizationLevel;
use crossbeam_queue::ArrayQueue;
use std::sync::Arc;

pub struct TextEmbedder {
    session_pool: Arc<ArrayQueue<Session>>,
    tokenizer: Tokenizer,
    pub dim: usize,
}

impl TextEmbedder {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Self {
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
        
        // We use a smaller pool (e.g., 2 or 3) but allow each to use full CPU.
        // This is usually better for Mac CPUs (Performance vs Efficiency cores).
        let num_sessions = 2; 
        let session_pool = Arc::new(ArrayQueue::new(num_sessions));

        for _ in 0..num_sessions {
            let session = Session::builder()
                .unwrap()
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .unwrap()
                // Removing with_intra_threads(1) to let ONNX use the optimal number of threads.
                .commit_from_file(model_path)
                .unwrap();
            session_pool.push(session).ok();
        }

        let mut embedder = Self { 
            session_pool, 
            tokenizer,
            dim: 0,
        };

        let dummy_vec = embedder.embed("test");
        embedder.dim = dummy_vec.len();

        embedder
    }

    pub fn embed(&self, text: &str) -> Vec<f32> {
        let encoding = self.tokenizer.encode(text, true).unwrap();
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let type_ids = encoding.get_type_ids();
        let seq_len = ids.len();

        let input_ids_array = Array2::from_shape_fn((1, seq_len), |(_, j)| ids[j] as i64);
        let attention_mask_array = Array2::from_shape_fn((1, seq_len), |(_, j)| mask[j] as i64);
        let token_type_ids_array = Array2::from_shape_fn((1, seq_len), |(_, j)| type_ids[j] as i64);

        let mut session = self.get_session();
        let outputs = session.run(ort::inputs![
            "input_ids" => Value::from_array(input_ids_array).unwrap(),
            "attention_mask" => Value::from_array(attention_mask_array).unwrap(),
            "token_type_ids" => Value::from_array(token_type_ids_array).unwrap(),
        ]).unwrap();

        let output_tensor = outputs["last_hidden_state"].try_extract_tensor::<f32>().unwrap();
        let (shape, data) = output_tensor;
        let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let view = ArrayViewD::from_shape(IxDyn(&shape_usize), data).unwrap();
        
        let pooled = view.mean_axis(Axis(1)).unwrap();
        let mut vector: Vec<f32> = pooled.index_axis(Axis(0), 0).iter().cloned().collect();
        
        drop(outputs);
        self.session_pool.push(session).ok();
        self.normalize(&mut vector);
        vector
    }

    fn get_session(&self) -> Session {
        loop {
            if let Some(s) = self.session_pool.pop() {
                return s;
            }
            std::thread::yield_now();
        }
    }

    fn normalize(&self, v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    pub fn embed_batch(&self, texts: Vec<String>) -> Vec<Vec<f32>> {
        if texts.is_empty() { return vec![]; }
        if texts.len() == 1 { return vec![self.embed(&texts[0])]; }

        let encodings = self.tokenizer.encode_batch(texts, true).unwrap();
        let batch_size = encodings.len();
        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);

        let mut input_ids = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask = Vec::with_capacity(batch_size * max_len);
        let mut token_type_ids = Vec::with_capacity(batch_size * max_len);

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let len = ids.len();
            input_ids.extend(ids.iter().map(|&id| id as i64));
            attention_mask.extend(encoding.get_attention_mask().iter().map(|&m| m as i64));
            token_type_ids.extend(encoding.get_type_ids().iter().map(|&t| t as i64));
            for _ in 0..(max_len - len) {
                input_ids.push(0);
                attention_mask.push(0);
                token_type_ids.push(0);
            }
        }

        let input_ids_array = Array2::from_shape_vec((batch_size, max_len), input_ids).unwrap();
        let attention_mask_array = Array2::from_shape_vec((batch_size, max_len), attention_mask).unwrap();
        let token_type_ids_array = Array2::from_shape_vec((batch_size, max_len), token_type_ids).unwrap();

        let mut session = self.get_session();
        let outputs = session.run(ort::inputs![
            "input_ids" => Value::from_array(input_ids_array).unwrap(),
            "attention_mask" => Value::from_array(attention_mask_array).unwrap(),
            "token_type_ids" => Value::from_array(token_type_ids_array).unwrap(),
        ]).unwrap();

        let output_tensor = outputs["last_hidden_state"].try_extract_tensor::<f32>().unwrap();
        let (shape, data) = output_tensor;
        let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let view = ArrayViewD::from_shape(IxDyn(&shape_usize), data).unwrap();
        
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let item_view = view.index_axis(Axis(0), i);
            let original_len = encodings[i].get_ids().len();
            let unpadded_item = item_view.slice(s![0..original_len, ..]);
            let pooled = unpadded_item.mean_axis(Axis(0)).unwrap();
            let mut vector: Vec<f32> = pooled.iter().cloned().collect();
            self.normalize(&mut vector);
            results.push(vector);
        }

        drop(outputs);
        self.session_pool.push(session).ok();
        results
    }
}
