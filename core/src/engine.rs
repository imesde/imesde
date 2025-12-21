use arc_swap::ArcSwapOption;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use crate::models::VectorRecord;

pub const NUM_SHARDS: usize = 16;
pub const SHARD_SIZE: usize = 1024;

pub struct Shard {
    buffer: Vec<ArcSwapOption<VectorRecord>>,
    index: AtomicUsize,
}

impl Shard {
    fn new(size: usize) -> Self {
        let mut buffer = Vec::with_capacity(size);
        for _ in 0..size {
            buffer.push(ArcSwapOption::from(None));
        }
        Self {
            buffer,
            index: AtomicUsize::new(0),
        }
    }

    fn insert(&self, record: Arc<VectorRecord>) {
        let pos = self.index.fetch_add(1, Ordering::SeqCst) % SHARD_SIZE;
        self.buffer[pos].store(Some(record));
    }
}

pub struct ShardedCircularBuffer {
    shards: Vec<Shard>,
}

impl ShardedCircularBuffer {
    pub fn new() -> Self {
        let mut shards = Vec::with_capacity(NUM_SHARDS);
        for _ in 0..NUM_SHARDS {
            shards.push(Shard::new(SHARD_SIZE));
        }
        Self { shards }
    }

    pub fn insert(&self, record: VectorRecord) {
        let shard_idx = self.get_shard_index(&record.id);
        self.shards[shard_idx].insert(Arc::new(record));
    }

    pub fn search(&self, query_vector: &[f32], k: usize) -> Vec<(Arc<VectorRecord>, f32)> {
        use crate::search::cosine_similarity;
        use rayon::prelude::*;
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;

        #[derive(Clone)]
        struct SearchResult {
            record: Arc<VectorRecord>,
            score: f32,
        }

        impl PartialEq for SearchResult {
            fn eq(&self, other: &Self) -> bool {
                self.score == other.score
            }
        }

        impl Eq for SearchResult {}

        impl PartialOrd for SearchResult {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                other.score.partial_cmp(&self.score)
            }
        }

        impl Ord for SearchResult {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        let heaps: Vec<BinaryHeap<SearchResult>> = self.shards
            .par_iter()
            .map(|shard| {
                let mut heap = BinaryHeap::with_capacity(k + 1);
                // We use sequential iteration inside the shard as it's only 1024 items,
                // but since we have 16 shards, we already use 16 threads.
                for slot in &shard.buffer {
                    if let Some(record) = slot.load_full() {
                        let score = cosine_similarity(query_vector, &record.vector);
                        heap.push(SearchResult { record, score });
                        if heap.len() > k {
                            heap.pop();
                        }
                    }
                }
                heap
            })
            .collect();

        let mut final_heap = BinaryHeap::with_capacity(k + 1);
        for heap in heaps {
            for result in heap {
                final_heap.push(result);
                if final_heap.len() > k {
                    final_heap.pop();
                }
            }
        }

        let mut results: Vec<_> = final_heap.into_iter()
            .map(|res| (res.record, res.score))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results
    }

    fn get_shard_index(&self, id: &str) -> usize {
        use std::hash::{Hash, Hasher};
        let mut hasher = fxhash::FxHasher::default();
        id.hash(&mut hasher);
        (hasher.finish() as usize) % NUM_SHARDS
    }
}