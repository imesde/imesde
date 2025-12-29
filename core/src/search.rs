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

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We checked that AVX2 is available.
            return unsafe { dot_product_avx2(v1, v2) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is standard on aarch64.
        return unsafe { dot_product_neon(v1, v2) };
    }

    #[cfg(not(target_arch = "aarch64"))]
    dot_product_scalar(v1, v2)
}

fn dot_product_scalar(v1: &[f32], v2: &[f32]) -> f32 {
    v1.iter().zip(v2).map(|(a, b)| a * b).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(v1: &[f32], v2: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let len = v1.len();
    let mut sum_vec = unsafe { _mm256_setzero_ps() };
    
    let mut i = 0;
    // Process 8 floats at a time
    while i + 8 <= len {
        unsafe {
            let a = _mm256_loadu_ps(v1.as_ptr().add(i));
            let b = _mm256_loadu_ps(v2.as_ptr().add(i));
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(a, b));
        }
        i += 8;
    }
    
    // Horizontal sum
    let sum128 = unsafe { _mm_add_ps(_mm256_castps256_ps128(sum_vec), _mm256_extractf128_ps(sum_vec, 1)) };
    let mut buf = [0.0; 4];
    unsafe { _mm_storeu_ps(buf.as_mut_ptr(), sum128) };
    let mut sum = buf.iter().sum::<f32>();

    // Handle remaining elements
    while i < len {
        unsafe {
            sum += *v1.get_unchecked(i) * *v2.get_unchecked(i);
        }
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_product_neon(v1: &[f32], v2: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let len = v1.len();
    let mut sum_vec = unsafe { vdupq_n_f32(0.0) };
    
    let mut i = 0;
    // Process 4 floats at a time
    while i + 4 <= len {
        unsafe {
            let a = vld1q_f32(v1.as_ptr().add(i));
            let b = vld1q_f32(v2.as_ptr().add(i));
            sum_vec = vfmaq_f32(sum_vec, a, b);
        }
        i += 4;
    }
    
    // Sum across the vector
    let mut sum = unsafe { vaddvq_f32(sum_vec) };
    
    // Handle remaining elements
    while i < len {
        unsafe {
            sum += *v1.get_unchecked(i) * *v2.get_unchecked(i);
        }
        i += 1;
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
