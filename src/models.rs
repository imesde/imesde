use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub timestamp: u64,
    pub metadata: String,
}

impl VectorRecord {
    pub fn new(id: String, vector: Vec<f32>, metadata: String) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_secs();

        Self {
            id,
            vector,
            timestamp,
            metadata,
        }
    }
}
