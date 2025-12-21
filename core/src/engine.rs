use arc_swap::ArcSwapOption;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use crate::models::VectorRecord;

pub const DEFAULT_NUM_SHARDS: usize = 16;
pub const DEFAULT_SHARD_SIZE: usize = 1024;

pub struct Shard {
    buffer: Vec<ArcSwapOption<VectorRecord>>,
    index: AtomicUsize,
    size: usize,
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
            size,
        }
    }

    fn insert(&self, record: Arc<VectorRecord>) {
        let pos = self.index.fetch_add(1, Ordering::SeqCst) % self.size;
        self.buffer[pos].store(Some(record));
    }
}

pub struct ShardedCircularBuffer {
    shards: Vec<Shard>,
    num_shards: usize,
}

impl ShardedCircularBuffer {
    pub fn new(num_shards: usize, shard_size: usize) -> Self {
        let mut shards = Vec::with_capacity(num_shards);
        for _ in 0..num_shards {
            shards.push(Shard::new(shard_size));
        }
        Self { shards, num_shards }
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
                let mut heap: BinaryHeap<SearchResult> = BinaryHeap::with_capacity(k + 1);
                // We use sequential iteration inside the shard as it's only 1024 items,
                // but since we have 16 shards, we already use 16 threads.
                for slot in &shard.buffer {
                    // OPTIMIZATION: Use load() instead of load_full().
                    // load() returns a Guard (temporary borrow) without incrementing the atomic ref count.
                    // This saves ~16,000 atomic operations per search query.
                    let guard = slot.load();
                    if let Some(record) = &*guard {
                        let score = cosine_similarity(query_vector, &record.vector);
                        
                        // We must clone the Arc only if we decide to keep it (Top-K)
                        // But since we are inside a heap loop, we can defer cloning until we are sure.
                        // However, BinaryHeap stores SearchResult which needs Arc<VectorRecord>.
                        // For the sake of the heap push, we clone. But this only happens if we beat the heap min.
                        // Optimization: First check score, then clone only if needed?
                        // Let's keep it simple: if heap is not full, push. If full and score > min, pop and push.
                        
                        // Peek at the smallest element in the min-heap (which is actually a max-heap reversed?)
                        // Wait, our Ord implementation is for Min-Heap behavior on scores? 
                        // Actually let's just push and pop. The cost of Arc clone is paid only K times + log(K) insertions.
                        // But wait, we need to push the STRUCT which holds the Arc.
                        
                        // To avoid cloning Arc for rejected items, we could check the score against heap.peek()
                        let should_push = if heap.len() < k {
                            true
                        } else if let Some(min_res) = heap.peek() {
                            // Our implementation: Ord is other.score.partial_cmp(&self.score)
                            // This means the "Greatest" element according to Ord is the one with the SMALLEST score.
                            // So heap.peek() returns the item with the LOWEST score (the one to evict).
                            // If current score > min_res.score, we should push.
                            score > min_res.score
                        } else {
                            true
                        };

                        if should_push {
                            // Only pay the atomic cost here!
                            let record_arc = Arc::clone(record);
                            heap.push(SearchResult { record: record_arc, score });
                            if heap.len() > k {
                                heap.pop();
                            }
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
        (hasher.finish() as usize) % self.num_shards
    }
}