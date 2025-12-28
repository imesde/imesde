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
    #[pyo3(signature = (model_path, tokenizer_path, num_shards=None, shard_size=None, track_centroid=None))]
    fn new(
        model_path: &str,
        tokenizer_path: &str,
        num_shards: Option<usize>,
        shard_size: Option<usize>,
        track_centroid: Option<bool>,
    ) -> PyResult<Self> {
        let ns = num_shards.unwrap_or(DEFAULT_NUM_SHARDS);
        let ss = shard_size.unwrap_or(DEFAULT_SHARD_SIZE);
        let tc = track_centroid.unwrap_or(true); // Default to enabled
        
        let embedder = Arc::new(TextEmbedder::new(model_path, tokenizer_path));
        let dim = embedder.dim;
        
        Ok(Self {
            buffer: Arc::new(ShardedCircularBuffer::new(ns, ss, dim, tc)),
            embedder,
            counter: Arc::new(AtomicUsize::new(0)),
        })
    }

    fn ingest(&self, py: Python<'_>, text: String) -> PyResult<f32> {
        let score = py.allow_threads(|| {
            let vector = self.embedder.embed(&text);
            let id = self.counter.fetch_add(1, Ordering::SeqCst);
            let record = VectorRecord::new(
                format!("log_{}", id),
                vector,
                text,
            );
            self.buffer.insert(record)
        });
        Ok(score)
    }

    fn ingest_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<f32>> {
        let scores = py.allow_threads(|| {
            use rayon::prelude::*;
            let scores: Vec<f32> = texts.par_iter().map(|text| {
                let vector = self.embedder.embed(text);
                let id = self.counter.fetch_add(1, Ordering::SeqCst);
                let record = VectorRecord::new(
                    format!("log_{}", id),
                    vector,
                    text.clone(),
                );
                self.buffer.insert(record)
            }).collect();
            scores
        });
        Ok(scores)
    }

    fn ingest_raw(&self, py: Python<'_>, vector: Vec<f32>, text: String) -> PyResult<f32> {
        let score = py.allow_threads(|| {
            let id = self.counter.fetch_add(1, Ordering::SeqCst);
            let record = VectorRecord::new(
                format!("log_{}", id),
                vector,
                text,
            );
            self.buffer.insert(record)
        });
        Ok(score)
    }

    fn ingest_batch_raw(&self, py: Python<'_>, vectors: Vec<Vec<f32>>, texts: Vec<String>) -> PyResult<Vec<f32>> {
        if vectors.len() != texts.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Vectors and texts must have the same length"));
        }
        let scores = py.allow_threads(|| {
            use rayon::prelude::*;
            vectors.into_par_iter().zip(texts.into_par_iter()).map(|(vector, text)| {
                let id = self.counter.fetch_add(1, Ordering::SeqCst);
                let record = VectorRecord::new(
                    format!("log_{}", id),
                    vector,
                    text,
                );
                self.buffer.insert(record)
            }).collect()
        });
        Ok(scores)
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

    fn get_centroid(&self) -> PyResult<Option<Vec<f32>>> {
        Ok(self.buffer.get_centroid())
    }

    fn get_outliers(&self, threshold: f32) -> PyResult<Vec<(String, f32)>> {
        let results = self.buffer.get_outliers(threshold);
        let py_results = results.into_iter()
            .map(|(record, score)| (record.metadata.clone(), score))
            .collect();
        Ok(py_results)
    }

    fn get_scores_from_centroid(&self) -> PyResult<Vec<(String, f32)>> {
        let results = self.buffer.get_scores_from_centroid();
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
