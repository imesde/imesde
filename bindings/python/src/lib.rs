use pyo3::prelude::*;
use ::imesde::engine::ShardedCircularBuffer;
use ::imesde::embedder::TextEmbedder;
use ::imesde::models::VectorRecord;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[pyclass]
struct PyImesde {
    buffer: Arc<ShardedCircularBuffer>,
    embedder: Arc<TextEmbedder>,
    counter: Arc<AtomicUsize>,
}

#[pymethods]
impl PyImesde {
    #[new]
    fn new(model_path: &str, tokenizer_path: &str) -> PyResult<Self> {
        Ok(Self {
            buffer: Arc::new(ShardedCircularBuffer::new()),
            embedder: Arc::new(TextEmbedder::new(model_path, tokenizer_path)),
            counter: Arc::new(AtomicUsize::new(0)),
        })
    }

    fn ingest(&self, text: &str) -> PyResult<()> {
        let vector = self.embedder.embed(text);
        let id = self.counter.fetch_add(1, Ordering::SeqCst);
        let record = VectorRecord::new(
            format!("log_{}", id),
            vector,
            text.to_string(),
        );
        self.buffer.insert(record);
        Ok(())
    }

    fn ingest_batch(&self, texts: Vec<String>) -> PyResult<()> {
        let chunk_size = 256; // Larger chunks for better ONNX utilization
        for chunk in texts.chunks(chunk_size) {
            let chunk_vec: Vec<String> = chunk.to_vec();
            let vectors = self.embedder.embed_batch(chunk_vec);
            
            for (i, vector) in vectors.into_iter().enumerate() {
                let id = self.counter.fetch_add(1, Ordering::SeqCst);
                let text = &chunk[i];
                let record = VectorRecord::new(
                    format!("log_{}", id),
                    vector,
                    text.clone(),
                );
                self.buffer.insert(record);
            }
        }
        Ok(())
    }

    fn search(&self, query: &str, k: usize) -> PyResult<Vec<(String, f32)>> {
        let query_vec = self.embedder.embed(query);
        let results = self.buffer.search(&query_vec, k);
        
        let py_results = results.into_iter()
            .map(|(record, score)| (record.metadata.clone(), score))
            .collect();
            
        Ok(py_results)
    }
}

#[pymodule]
fn imesde(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyImesde>()?;
    Ok(())
}
