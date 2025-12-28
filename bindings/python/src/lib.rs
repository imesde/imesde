use pyo3::prelude::*;
use ::imesde::engine::{ShardedCircularBuffer, DEFAULT_NUM_SHARDS, DEFAULT_SHARD_SIZE};
use ::imesde::embedder::TextEmbedder;
use ::imesde::mamba::MambaWorker;
use ::imesde::models::VectorRecord;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

enum EmbedderBackend {
    Onnx(Arc<TextEmbedder>),
    Mamba(Arc<MambaWorker>), // Rimosso Mutex perché DashMap gestisce la concorrenza interna
}

#[pyclass]
struct PyImesde {
    buffer: Arc<ShardedCircularBuffer>,
    embedder: EmbedderBackend,
    counter: Arc<AtomicUsize>,
}

#[pymethods]
impl PyImesde {
    #[new]
    #[pyo3(signature = (model_path, tokenizer_path, num_shards=None, shard_size=None, use_mamba=false))]
    fn new(
        model_path: &str,
        tokenizer_path: &str,
        num_shards: Option<usize>,
        shard_size: Option<usize>,
        use_mamba: bool,
    ) -> PyResult<Self> {
        let ns = num_shards.unwrap_or(DEFAULT_NUM_SHARDS);
        let ss = shard_size.unwrap_or(DEFAULT_SHARD_SIZE);
        
        let embedder = if use_mamba {
            let model_id = if model_path.is_empty() { None } else { Some(model_path) };
            let worker = MambaWorker::new(model_id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            EmbedderBackend::Mamba(Arc::new(worker))
        } else {
            let emb = TextEmbedder::new(model_path, tokenizer_path);
            EmbedderBackend::Onnx(Arc::new(emb))
        };

        Ok(Self {
            buffer: Arc::new(ShardedCircularBuffer::new(ns, ss)),
            embedder,
            counter: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Ingestione batch con supporto a Rayon e stati Mamba isolati
    #[pyo3(signature = (texts, keys=None))]
    fn ingest_batch(&self, py: Python<'_>, texts: Vec<String>, keys: Option<Vec<String>>) -> PyResult<()> {
        py.allow_threads(|| {
            use rayon::prelude::*;
            
            match &self.embedder {
                EmbedderBackend::Onnx(emb) => {
                    let chunk_size = 128;
                    texts.par_chunks(chunk_size).for_each(|chunk| {
                        let vectors = emb.embed_batch(chunk.to_vec());
                        for (i, vector) in vectors.into_iter().enumerate() {
                            let id = self.counter.fetch_add(1, Ordering::SeqCst);
                            self.buffer.insert(VectorRecord::new(format!("log_{}", id), vector, chunk[i].clone()));
                        }
                    });
                },
                EmbedderBackend::Mamba(worker) => {
                    // Qui usiamo Rayon sui voli! 
                    // Se keys è None, usiamo None come chiave (stato globale)
                    let keys_vec = keys.unwrap_or_else(|| vec!["global".to_string(); texts.len()]);
                    
                    texts.into_par_iter().zip(keys_vec.into_par_iter()).for_each(|(text, key)| {
                        if let Ok(vector) = worker.embed_stateful(&text, Some(&key)) {
                            let id = self.counter.fetch_add(1, Ordering::SeqCst);
                            self.buffer.insert(VectorRecord::new(format!("log_{}", id), vector, text));
                        }
                    });
                }
            }
        });
        Ok(())
    }

    fn ingest(&self, py: Python<'_>, text: String, key: Option<String>) -> PyResult<()> {
        let vector = match &self.embedder {
            EmbedderBackend::Onnx(emb) => emb.embed(&text),
            EmbedderBackend::Mamba(worker) => {
                worker.embed_stateful(&text, key.as_deref())
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            }
        };
        
        let id = self.counter.fetch_add(1, Ordering::SeqCst);
        self.buffer.insert(VectorRecord::new(format!("log_{}", id), vector, text));
        Ok(())
    }

    fn search(&self, py: Python<'_>, query: String, k: usize) -> PyResult<Vec<(String, f32)>> {
        let query_vec = match &self.embedder {
            EmbedderBackend::Onnx(emb) => emb.embed(&query),
            EmbedderBackend::Mamba(worker) => {
                worker.embed_stateful(&query, None)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            }
        };
        
        let results = self.buffer.search(&query_vec, k);
        Ok(results.into_iter().map(|(r, s)| (r.metadata.clone(), s)).collect())
    }

    fn search_raw(&self, _py: Python<'_>, query_vec: Vec<f32>, k: usize) -> PyResult<Vec<(String, f32)>> {
        let results = self.buffer.search(&query_vec, k);
        Ok(results.into_iter().map(|(r, s)| (r.metadata.clone(), s)).collect())
    }

    fn embed_query(&self, py: Python<'_>, text: String) -> PyResult<Vec<f32>> {
        match &self.embedder {
            EmbedderBackend::Onnx(emb) => Ok(emb.embed(&text)),
            EmbedderBackend::Mamba(worker) => {
                worker.embed_stateful(&text, None)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            }
        }
    }
}

#[pymodule]
fn imesde(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyImesde>()?;
    Ok(())
}
