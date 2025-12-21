pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    if v1.len() != v2.len() || v1.is_empty() {
        return 0.0;
    }

    let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm_v1: f32 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let norm_v2: f32 = v2.iter().map(|b| b * b).sum::<f32>().sqrt();

    if norm_v1 == 0.0 || norm_v2 == 0.0 {
        return 0.0;
    }

    dot_product / (norm_v1 * norm_v2)
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
