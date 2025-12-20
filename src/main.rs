use std::sync::Arc;
use imesde::models::VectorRecord;
use imesde::engine::{ShardedCircularBuffer, NUM_SHARDS};

fn main() {
    let buffer = Arc::new(ShardedCircularBuffer::new());
    println!("Lock-free sharded circular buffer initialized with {} shards.", NUM_SHARDS);

    let record = VectorRecord::new(
        "doc_init".to_string(),
        vec![1.0, 2.0, 3.0],
        "initialization".to_string(),
    );
    buffer.insert(record);
    println!("Initial record inserted.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_stress_sharded_buffer() {
        let buffer = Arc::new(ShardedCircularBuffer::new());
        let num_threads = 10;
        let inserts_per_thread = 1000;
        let mut handles = Vec::new();

        println!("Starting stress test with {} threads...", num_threads);

        for t in 0..num_threads {
            let buffer_ref = Arc::clone(&buffer);
            let handle = thread::spawn(move || {
                for i in 0..inserts_per_thread {
                    let record = VectorRecord::new(
                        format!("thread_{}_doc_{}", t, i),
                        vec![i as f32; 128],
                        format!("metadata from thread {}", t),
                    );
                    buffer_ref.insert(record);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        println!("Test Passed: The circular buffer works under stress!");
    }

    #[test]
    fn test_search_functionality() {
        let buffer = ShardedCircularBuffer::new();
        
        // Insert some known vectors
        buffer.insert(VectorRecord::new("1".into(), vec![1.0, 0.0, 0.0], "vec 1".into()));
        buffer.insert(VectorRecord::new("2".into(), vec![0.0, 1.0, 0.0], "vec 2".into()));
        buffer.insert(VectorRecord::new("3".into(), vec![0.5, 0.5, 0.0], "vec 3".into()));

        let query = vec![1.0, 0.1, 0.0];
        let top_k = buffer.search(&query, 2);

        assert_eq!(top_k.len(), 2);
        assert_eq!(top_k[0].0.id, "1"); // Most similar
        assert_eq!(top_k[1].0.id, "3"); // Second most similar
        
        println!("Search test passed: Found records {:?} with similarities {:?}", 
            top_k.iter().map(|(r, _)| &r.id).collect::<Vec<_>>(),
            top_k.iter().map(|(_, s)| s).collect::<Vec<_>>()
        );
    }
}
