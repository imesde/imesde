use tokenizers::Tokenizer;
use ndarray::{Array2, Axis, ArrayViewD, IxDyn, s};
use std::cell::RefCell;

thread_local! {
    static SESSION: RefCell<Option<ort::session::Session>> = RefCell::new(None);
}

pub struct TextEmbedder {
    model_path: String,
    tokenizer: Tokenizer,
    pub dim: usize,
}

impl TextEmbedder {
    pub fn new(model_path: &str, tokenizer_path: &str) -> Self {
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
        
        // Initial session just to get the dimension
        let _temp_session = ort::session::Session::builder()
            .unwrap()
            .commit_from_file(model_path)
            .unwrap();

        let mut embedder = Self { 
            model_path: model_path.to_string(),
            tokenizer,
            dim: 0,
        };

        // Perform a dry run to auto-detect dimension
        let dummy_vec = embedder.embed("test");
        embedder.dim = dummy_vec.len();

        embedder
    }

    fn with_session<F, R>(&self, f: F) -> R 
    where F: FnOnce(&mut ort::session::Session) -> R 
    {
        SESSION.with(|cell| {
            let mut maybe_session = cell.borrow_mut();
            if maybe_session.is_none() {
                let session = ort::session::Session::builder()
                    .unwrap()
                    .commit_from_file(&self.model_path)
                    .unwrap();
                *maybe_session = Some(session);
            }
            f(maybe_session.as_mut().unwrap())
        })
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

        let input_ids_val = ort::value::Value::from_array(input_ids_array).unwrap();
        let attention_mask_val = ort::value::Value::from_array(attention_mask_array).unwrap();
        let token_type_ids_val = ort::value::Value::from_array(token_type_ids_array).unwrap();

        let (shape, data) = self.with_session(|session| {
            let outputs = session.run(ort::inputs![
                "input_ids" => input_ids_val,
                "attention_mask" => attention_mask_val,
                "token_type_ids" => token_type_ids_val,
            ]).unwrap();
            let output_tensor = outputs["last_hidden_state"].try_extract_tensor::<f32>().unwrap();
            let (shape, data) = output_tensor;
            (shape.to_vec(), data.to_vec())
        });
        
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let view = ArrayViewD::from_shape(IxDyn(&dims), &data).unwrap();
        
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

    pub fn embed_batch(&self, texts: Vec<String>) -> Vec<Vec<f32>> {
        if texts.is_empty() {
            return vec![];
        }

        let encodings = self.tokenizer.encode_batch(texts, true).unwrap();
        let batch_size = encodings.len();
        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);

        let mut input_ids = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask = Vec::with_capacity(batch_size * max_len);
        let mut token_type_ids = Vec::with_capacity(batch_size * max_len);

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();
            let len = ids.len();

            input_ids.extend(ids.iter().map(|&id| id as i64));
            attention_mask.extend(mask.iter().map(|&m| m as i64));
            token_type_ids.extend(type_ids.iter().map(|&t| t as i64));

            for _ in 0..(max_len - len) {
                input_ids.push(0);
                attention_mask.push(0);
                token_type_ids.push(0);
            }
        }

        let input_ids_array = Array2::from_shape_vec((batch_size, max_len), input_ids).unwrap();
        let attention_mask_array = Array2::from_shape_vec((batch_size, max_len), attention_mask).unwrap();
        let token_type_ids_array = Array2::from_shape_vec((batch_size, max_len), token_type_ids).unwrap();

        let input_ids_val = ort::value::Value::from_array(input_ids_array).unwrap();
        let attention_mask_val = ort::value::Value::from_array(attention_mask_array).unwrap();
        let token_type_ids_val = ort::value::Value::from_array(token_type_ids_array).unwrap();

        let (shape, data) = self.with_session(|session| {
            let outputs = session.run(ort::inputs![
                "input_ids" => input_ids_val,
                "attention_mask" => attention_mask_val,
                "token_type_ids" => token_type_ids_val,
            ]).unwrap();
            let output_tensor = outputs["last_hidden_state"].try_extract_tensor::<f32>().unwrap();
            let (shape, data) = output_tensor;
            (shape.to_vec(), data.to_vec())
        });
        
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let view = ArrayViewD::from_shape(IxDyn(&dims), &data).unwrap();
        
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let item_view = view.index_axis(Axis(0), i);
            let original_len = encodings[i].get_ids().len();
            
            let unpadded_item = item_view.slice(s![0..original_len, ..]);
            let pooled = unpadded_item.mean_axis(Axis(0)).unwrap();
            
            let mut vector: Vec<f32> = pooled.iter().cloned().collect();
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > f32::EPSILON {
                for x in vector.iter_mut() {
                    *x /= norm;
                }
            }
            results.push(vector);
        }

        results
    }
}
