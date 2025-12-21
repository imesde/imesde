use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::path::Path;
use std::io::{self, BufRead, Write};
use std::fs::File;
use std::thread;

use imesde::models::VectorRecord;
use imesde::engine::ShardedCircularBuffer;
use imesde::embedder::TextEmbedder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Core Initialization
    let buffer = Arc::new(ShardedCircularBuffer::new());
    let log_count = Arc::new(AtomicUsize::new(0));

    // 2. AI Initialization
    let model_path = "model/model.onnx";
    let tokenizer_path = "model/tokenizer.json";

    if !Path::new(model_path).exists() || !Path::new(tokenizer_path).exists() {
        eprintln!("❌ Error: Model files not found. Place them in 'model/' directory.");
        std::process::exit(1);
    }

    let embedder = Arc::new(TextEmbedder::new(model_path, tokenizer_path));
    println!("🚀 Imesde Engine & AI Ready (Dim: {}).", embedder.dim);
    println!("📝 Commands: /search <query>, /status, /exit");
    println!("--------------------------------------------------");

    // 3. Background Ingestion Thread
    let buffer_ingest = Arc::clone(&buffer);
    let embedder_ingest = Arc::clone(&embedder);
    let count_ingest = Arc::clone(&log_count);

    thread::spawn(move || {
        let stdin = io::stdin();
        let mut reader = stdin.lock();
        let mut line = String::new();

        while let Ok(n) = reader.read_line(&mut line) {
            if n == 0 { break; } // EOF
            
            let text = line.trim();
            if !text.is_empty() {
                // Generate embedding and insert
                let vector = embedder_ingest.embed(text);
                let current_id = count_ingest.fetch_add(1, Ordering::SeqCst);
                
                let record = VectorRecord::new(
                    format!("log_{}", current_id),
                    vector,
                    text.to_string(),
                );
                buffer_ingest.insert(record);
            }
            line.clear();
        }
    });

    // 4. Interactive UI Loop (Using /dev/tty to keep stdin free for pipes)
    let mut tty_reader = io::BufReader::new(File::open("/dev/tty")?);
    let mut input = String::new();

    loop {
        print!("imesde> ");
        io::stdout().flush()?;
        
        input.clear();
        if tty_reader.read_line(&mut input)? == 0 { break; }
        
        let cmd = input.trim();
        if cmd.is_empty() { continue; }

        if cmd.starts_with("/search ") {
            let query = &cmd[8..];
            println!("🔍 Searching for: '{}'...", query);
            
            let query_vec = embedder.embed(query);
            let results = buffer.search(&query_vec, 5);

            if results.is_empty() {
                println!("   No records found yet.");
            } else {
                for (record, score) in results {
                    println!("   - [{:.4}] {}", score, record.metadata);
                }
            }
        } else if cmd == "/status" {
            let total = log_count.load(Ordering::SeqCst);
            println!("📊 Status: {} logs ingested in circular buffer.", total);
        } else if cmd == "/exit" {
            println!("👋 Goodbye!");
            break;
        } else {
            println!("❓ Unknown command. Use /search <query>, /status or /exit");
        }
    }

    Ok(())
}
