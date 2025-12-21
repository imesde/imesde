use pyo3::prelude::*;
use ::imesde::engine::{ShardedCircularBuffer, DEFAULT_NUM_SHARDS, DEFAULT_SHARD_SIZE};
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
    #[pyo3(signature = (model_path, tokenizer_path, num_shards=None, shard_size=None))]
    fn new(
        model_path: &str,
        tokenizer_path: &str,
        num_shards: Option<usize>,
        shard_size: Option<usize>,
    ) -> PyResult<Self> {
        let ns = num_shards.unwrap_or(DEFAULT_NUM_SHARDS);
        let ss = shard_size.unwrap_or(DEFAULT_SHARD_SIZE);
        Ok(Self {
            buffer: Arc::new(ShardedCircularBuffer::new(ns, ss)),
            embedder: Arc::new(TextEmbedder::new(model_path, tokenizer_path)),
            counter: Arc::new(AtomicUsize::new(0)),
        })
    }

    fn ingest(&self, py: Python<'_>, text: String) -> PyResult<()> {
        py.allow_threads(|| {
            let vector = self.embedder.embed(&text);
            let id = self.counter.fetch_add(1, Ordering::SeqCst);
            let record = VectorRecord::new(
                format!("log_{}", id),
                vector,
                text,
            );
            self.buffer.insert(record);
        });
        Ok(())
    }

    fn ingest_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<()> {
        py.allow_threads(|| {
            use rayon::prelude::*;
            let chunk_size = 128; // Larger chunks to better utilize each session's internal threads
            texts.par_chunks(chunk_size).for_each(|chunk| {
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
            });
        });
        Ok(())
    }

    fn search(&self, py: Python<'_>, query: String, k: usize) -> PyResult<Vec<(String, f32)>> {
        let results = py.allow_threads(|| {
            let query_vec = self.embedder.embed(&query);
            self.buffer.search(&query_vec, k)
        });
        
        let py_results = results.into_iter()
            .map(|(record, score)| (record.metadata.clone(), score))
            .collect();
            
        Ok(py_results)
    }

    fn embed_query(&self, py: Python<'_>, text: String) -> PyResult<Vec<f32>> {
        let vector = py.allow_threads(|| {
            self.embedder.embed(&text)
        });
        Ok(vector)
    }

    fn search_raw(&self, query_vector: Vec<f32>, k: usize) -> PyResult<Vec<(String, f32)>> {
        let results = self.buffer.search(&query_vector, k);
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
