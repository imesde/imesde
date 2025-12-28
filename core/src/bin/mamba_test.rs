use imesde::mamba::MambaWorker;
use std::time::Instant;
use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let model_path = if args.len() > 1 {
        Some(args[1].as_str())
    } else {
        None
    };

    println!("🚀 Initializing Mamba Model...");
    let start = Instant::now();
    
    let mamba = MambaWorker::new(model_path)?;
    
    println!("✅ Model loaded in {:.2?}", start.elapsed());

    let text = "Il semantic radar sta scansionando il flusso di dati.";
    println!("📝 Processing text: '{}'", text);

    let start_embed = Instant::now();
    let embedding = mamba.embed(text)?;
    
    println!("⚡ Inference time: {:.2?}", start_embed.elapsed());
    println!("📊 Embedding Dimensions: {}", embedding.len());
    println!("🔢 First 5 values: {:?}", &embedding[0..5]);

    Ok(())
}