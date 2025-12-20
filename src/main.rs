use std::sync::Arc;
use std::path::Path;
use std::io::{self, BufRead};
use imesde::models::VectorRecord;
use imesde::engine::ShardedCircularBuffer;
use imesde::embedder::TextEmbedder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Engine Initialization
    let buffer = Arc::new(ShardedCircularBuffer::new());
    
    // 2. AI Initialization
    let model_path = "model/model.onnx";
    let tokenizer_path = "model/tokenizer.json";

    if !Path::new(model_path).exists() || !Path::new(tokenizer_path).exists() {
        eprintln!("❌ Error: Model files not found. Please place them in the 'model/' directory.");
        std::process::exit(1);
    }

    let embedder = Arc::new(TextEmbedder::new(model_path, tokenizer_path));
    println!("🚀 Imesde Engine & AI Ready (Detected Dimension: {}).", embedder.dim);

    // 3. Real-time Ingestion Pipeline (STDIN)
    // Checking if we have piped input (e.g., cat logs.txt | imesde)
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    
    println!("📥 Listening to stream (stdin)... Press Ctrl+C to stop or send search query.");

    let mut line = String::new();
    let mut count = 0;

    // We can use the embedder in multiple threads safely now
    while reader.read_line(&mut line)? > 0 {
        let text = line.trim();
        if text.is_empty() {
            line.clear();
            continue;
        }

        // Check if it's a special search command for demo purposes: "/search query"
        if text.starts_with("/search ") {
            let query = &text[8..];
            println!("\n🔍 Semantic Search for: '{}'", query);
            let query_vec = embedder.embed(query);
            let results = buffer.search(&query_vec, 3);
            
            for (record, score) in results {
                println!("   - [{:.4}] {}", score, record.metadata);
            }
            println!();
        } else {
            // Normal ingestion
            let vector = embedder.embed(text);
            let record = VectorRecord::new(
                format!("log_{}", count),
                vector,
                text.to_string(),
            );
            buffer.insert(record);
            count += 1;
            
            if count % 1 == 0 {
                println!("✨ Ingested {} records...", count);
            }
        }
        
        line.clear();
    }

    Ok(())
}