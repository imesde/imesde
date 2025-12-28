use std::sync::Arc;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use rayon::prelude::*;
use crate::models::VectorRecord;
use crate::engine::Shard;

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

pub fn perform_search(shards: &[Shard], query_vector: &[f32], k: usize) -> Vec<(Arc<VectorRecord>, f32)> {
    let heaps: Vec<BinaryHeap<SearchResult>> = shards
        .par_iter()
        .map(|shard| {
            let mut heap: BinaryHeap<SearchResult> = BinaryHeap::with_capacity(k + 1);
            for slot in &shard.buffer {
                let guard = slot.load();
                if let Some(record) = &*guard {
                    let score = cosine_similarity(query_vector, &record.vector);
                    let should_push = if heap.len() < k {
                        true
                    } else if let Some(min_res) = heap.peek() {
                        score > min_res.score
                    } else {
                        true
                    };

                    if should_push {
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

/// Generic helper to scan all shards and apply a filter/map operation.
fn scan_shards<F>(shards: &[Shard], filter_map: F) -> Vec<(Arc<VectorRecord>, f32)>
where
    F: Fn(Arc<VectorRecord>) -> Option<(Arc<VectorRecord>, f32)> + Sync + Send,
{
    shards
        .par_iter()
        .flat_map(|shard| {
            let mut results = Vec::new();
            for slot in &shard.buffer {
                let guard = slot.load();
                if let Some(record) = &*guard {
                    if let Some(res) = filter_map(Arc::clone(record)) {
                        results.push(res);
                    }
                }
            }
            results
        })
        .collect()
}

pub fn detect_outliers(shards: &[Shard], centroid: &[f32], threshold: f32) -> Vec<(Arc<VectorRecord>, f32)> {
    scan_shards(shards, |record| {
        let similarity = cosine_similarity(centroid, &record.vector);
        if similarity < threshold {
            Some((record, similarity))
        } else {
            None
        }
    })
}

pub fn compute_centroid_scores(shards: &[Shard], centroid: &[f32]) -> Vec<(Arc<VectorRecord>, f32)> {
    scan_shards(shards, |record| {
        let similarity = cosine_similarity(centroid, &record.vector);
        Some((record, similarity))
    })
}

pub fn dot_product(v1: &[f32], v2: &[f32]) -> f32 {
    let len = v1.len();
    if len != v2.len() || len == 0 {
        return 0.0;
    }

    // Manual unrolling/hinting often helps the compiler verify purely safe access
    // isn't bound-checked inside the hot loop.
    let mut sum = 0.0;
    for i in 0..len {
        // SAFETY: We checked lengths are equal.
        // Using get_unchecked would be unsafe but faster. 
        // For now, simple indexing allows auto-vectorization if compiled with -C target-cpu=native
        sum += v1[i] * v2[i];
    }
    sum
}

// Since our vectors are already normalized in the embedder, 
// cosine similarity is just the dot product.
pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    dot_product(v1, v2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&v1, &v2);
        assert!((sim - 1.0).abs() < f32::EPSILON);

        let v3 = vec![0.0, 1.0, 0.0];
        let sim_ortho = cosine_similarity(&v1, &v3);
        assert!(sim_ortho.abs() < f32::EPSILON);

        let v4 = vec![-1.0, 0.0, 0.0];
        let sim_opp = cosine_similarity(&v1, &v4);
        assert!((sim_opp + 1.0).abs() < f32::EPSILON);
    }
}
