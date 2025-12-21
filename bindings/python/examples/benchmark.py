import imesde
import time
import statistics
import platform
import gc
import concurrent.futures

# Configuration
MODEL_PATH = "model/model.onnx"
TOKENIZER_PATH = "model/tokenizer.json"
NUM_RECORDS = 5000     # Increased to 5k to make the search work a bit harder
SEARCH_ITERATIONS = 5000 # More iterations for stable microsecond measurement
CONCURRENT_THREADS = 4

def get_system_info():
    try:
        return f"{platform.system()} {platform.release()} ({platform.machine()})"
    except:
        return "Unknown System"

def run_benchmark():
    print(f"\n🚀 imesde Performance Benchmark")
    print(f"💻 System:  {get_system_info()}")
    print(f"📊 Dataset: {NUM_RECORDS} records")
    print("-" * 60)

    # 1. Initialization
    db = imesde.PyImesde(MODEL_PATH, TOKENIZER_PATH)
    texts = [f"Financial market update log entry number {i} regarding inflation." for i in range(NUM_RECORDS)]

    # --- TEST 1: BATCH INGESTION ---
    print(f"\n[1/4] 🚀 Ingesting Data...")
    start_time = time.perf_counter()
    db.ingest_batch(texts)
    end_time = time.perf_counter()
    print(f"   ⏱️  Time:       {end_time - start_time:.2f} s")

    # --- TEST 2: AI LATENCY (Embedding Only) ---
    print(f"\n[2/4] 🧠 Measuring AI Embedding Latency (CPU/ONNX)...")
    # Warmup
    _ = db.embed_query("warmup")
    
    embed_latencies = []
    start_embed = time.perf_counter()
    for _ in range(100): # Run 100 times
        t0 = time.perf_counter_ns()
        _ = db.embed_query("test query for embedding speed")
        t1 = time.perf_counter_ns()
        embed_latencies.append((t1 - t0) / 1000.0)
    
    avg_embed = statistics.mean(embed_latencies)
    print(f"   ⏱️  Avg Embedding: {avg_embed:.2f} μs ({avg_embed/1000:.2f} ms)")
    print(f"   ℹ️  This is the cost of the Neural Network.")

    # --- TEST 3: ENGINE LATENCY (Pure Search) ---
    print(f"\n[3/4] ⚡ Measuring Engine Search Latency (Pure Vector Search)...")
    # Pre-calculate a vector to skip embedding cost during loop
    query_vec = db.embed_query("inflation analysis")
    
    search_latencies = []
    start_search = time.perf_counter()
    for _ in range(SEARCH_ITERATIONS):
        t0 = time.perf_counter_ns()
        _ = db.search_raw(query_vec, k=5)
        t1 = time.perf_counter_ns()
        search_latencies.append((t1 - t0) / 1000.0)
    
    avg_search = statistics.mean(search_latencies)
    p99_search = statistics.quantiles(search_latencies, n=100)[98]
    
    print(f"   ⏱️  Avg Search:    {avg_search:.2f} μs")
    print(f"   ⏱️  P99 Search:    {p99_search:.2f} μs")
    print(f"   🚀 Engine OPS:    {1_000_000 / avg_search:.0f} queries/sec")

    # --- TEST 4: END-TO-END CONCURRENCY ---
    print(f"\n[4/4] 🌐 Testing Concurrent End-to-End Search ({CONCURRENT_THREADS} threads)...")
    
    def search_worker(n):
        for _ in range(n):
            db.search("concurrency test", k=5)

    iters_per_thread = 200 # Total 800 searches
    
    gc.collect()
    start_conc = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_THREADS) as executor:
        futures = [executor.submit(search_worker, iters_per_thread) for _ in range(CONCURRENT_THREADS)]
        concurrent.futures.wait(futures)
    end_conc = time.perf_counter()
    
    conc_qps = (iters_per_thread * CONCURRENT_THREADS) / (end_conc - start_conc)
    print(f"   ⚡ Total QPS:      {conc_qps:.0f} queries/sec (limited by embedding)")
    print("-" * 60)

if __name__ == "__main__":
    run_benchmark()