pub fn dot_product(v1: &[f32], v2: &[f32]) -> f32 {
    if v1.len() != v2.len() || v1.is_empty() {
        return 0.0;
    }

    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
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
