use arc_swap::ArcSwapOption;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::RwLock;
use crate::models::VectorRecord;

pub const DEFAULT_NUM_SHARDS: usize = 16;
pub const DEFAULT_SHARD_SIZE: usize = 1024;

pub struct Shard {
    pub(crate) buffer: Vec<ArcSwapOption<VectorRecord>>,
    pub(crate) index: AtomicUsize,
    pub(crate) size: usize,
    pub(crate) sum: RwLock<Vec<f32>>,
}

impl Shard {
    fn new(size: usize, dim: usize) -> Self {
        let mut buffer = Vec::with_capacity(size);
        for _ in 0..size {
            buffer.push(ArcSwapOption::from(None));
        }
        Self {
            buffer,
            index: AtomicUsize::new(0),
            size,
            sum: RwLock::new(vec![0.0; dim]),
        }
    }
}

pub struct ShardedCircularBuffer {
    shards: Vec<Shard>,
    num_shards: usize,
    global_count: AtomicUsize,
    track_centroid: bool,
    dim: usize,
}

impl ShardedCircularBuffer {
    pub fn new(num_shards: usize, shard_size: usize, dim: usize, track_centroid: bool) -> Self {
        let mut shards = Vec::with_capacity(num_shards);
        for _ in 0..num_shards {
            shards.push(Shard::new(shard_size, dim));
        }
        Self { 
            shards, 
            num_shards, 
            global_count: AtomicUsize::new(0),
            track_centroid,
            dim,
        }
    }

    fn get_global_sum(&self) -> Vec<f32> {
        let mut global_sum = vec![0.0; self.dim];
        for shard in &self.shards {
            let shard_sum = shard.sum.read().unwrap();
            for (i, v) in shard_sum.iter().enumerate() {
                global_sum[i] += v;
            }
        }
        global_sum
    }

    pub fn insert(&self, record: VectorRecord) -> f32 {
        use crate::search::cosine_similarity;

        // 1. Calculate instant anomaly score
        let mut score = 1.0;
        if self.track_centroid {
            let count = self.global_count.load(Ordering::SeqCst);
            if count > 0 {
                let mut mean = self.get_global_sum();
                for v in mean.iter_mut() {
                    *v /= count as f32;
                }
                score = cosine_similarity(&mean, &record.vector);
            }
        }

        let shard_idx = self.get_shard_index(&record.id);
        let shard = &self.shards[shard_idx];
        let pos = shard.index.fetch_add(1, Ordering::SeqCst) % shard.size;

        if self.track_centroid {
            // Update with tracking: needs record.clone() for the global sum update
            let new_record_arc = Arc::new(record.clone());
            let old_record = shard.buffer[pos].swap(Some(new_record_arc));

            let mut shard_sum = shard.sum.write().unwrap();
            if let Some(old) = old_record {
                for (i, v) in old.vector.iter().enumerate() {
                    shard_sum[i] -= v;
                }
            } else {
                self.global_count.fetch_add(1, Ordering::SeqCst);
            }
            for (i, v) in record.vector.iter().enumerate() {
                shard_sum[i] += v;
            }
        } else {
            // Faster update: no tracking, no clone, move record directly
            let old_record = shard.buffer[pos].swap(Some(Arc::new(record)));
            if old_record.is_none() {
                self.global_count.fetch_add(1, Ordering::SeqCst);
            }
        }

        score
    }

    pub fn search(&self, query_vector: &[f32], k: usize) -> Vec<(Arc<VectorRecord>, f32)> {
        crate::search::perform_search(&self.shards, query_vector, k)
    }

    pub fn get_centroid(&self) -> Option<Vec<f32>> {
        let count = self.global_count.load(Ordering::SeqCst);
        
        if count > 0 {
            let mut mean = self.get_global_sum();
            for v in mean.iter_mut() {
                *v /= count as f32;
            }
            return Some(mean);
        }
        None
    }

    pub fn get_outliers(&self, threshold: f32) -> Vec<(Arc<VectorRecord>, f32)> {
        let centroid = match self.get_centroid() {
            Some(c) => c,
            None => return vec![],
        };
        crate::search::detect_outliers(&self.shards, &centroid, threshold)
    }

    pub fn get_scores_from_centroid(&self) -> Vec<(Arc<VectorRecord>, f32)> {
        let centroid = match self.get_centroid() {
            Some(c) => c,
            None => return vec![],
        };
        crate::search::compute_centroid_scores(&self.shards, &centroid)
    }

    fn get_shard_index(&self, id: &str) -> usize {
        use std::hash::{Hash, Hasher};
        let mut hasher = fxhash::FxHasher::default();
        id.hash(&mut hasher);
        (hasher.finish() as usize) % self.num_shards
    }
}
