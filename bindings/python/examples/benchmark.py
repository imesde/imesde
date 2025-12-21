import imesde
import time
import statistics
import platform
import gc
import concurrent.futures

# Configuration
MODEL_PATH = "model/model.onnx"
TOKENIZER_PATH = "model/tokenizer.json"
NUM_RECORDS = 2000     # Increased to 2k to better stress the engine
SEARCH_ITERATIONS = 1000
CONCURRENT_THREADS = 4 # Simulate 4 concurrent users/threads searching

def get_system_info():
    try:
        return f"{platform.system()} {platform.release()} ({platform.machine()})"
    except:
        return "Unknown System"

def run_benchmark():
    print(f"\n🚀 imesde Performance Benchmark v2.0")
    print(f"💻 System:  {get_system_info()}")
    print(f"📊 Dataset: {NUM_RECORDS} records")
    print("-" * 60)

    # 1. Initialization
    try:
        db = imesde.PyImesde(MODEL_PATH, TOKENIZER_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    texts = [f"Financial market update log entry number {i} regarding inflation, rates and tech stocks." for i in range(NUM_RECORDS)]

    # --- TEST 1: SINGLE INGESTION ---
    print(f"\n[1/4] 🟢 Testing Single Ingestion (Baseline)...")
    gc.collect()
    start_time = time.perf_counter()
    for t in texts:
        db.ingest(t)
    end_time = time.perf_counter()
    
    single_time = end_time - start_time
    single_ips = NUM_RECORDS / single_time
    print(f"   ⏱️  Time:       {single_time:.2f} s")
    print(f"   ⚡ Throughput: {single_ips:.0f} items/sec")

    # --- TEST 2: BATCH INGESTION ---
    print(f"\n[2/4] 🚀 Testing Parallel Batch Ingestion...")
    # Note: This appends to the buffer, effectively doubling the data if we don't restart,
    # but circular buffer handles it.
    gc.collect()
    start_time = time.perf_counter()
    db.ingest_batch(texts)
    end_time = time.perf_counter()
    
    batch_time = end_time - start_time
    batch_ips = NUM_RECORDS / batch_time
    speedup = single_time / batch_time
    
    print(f"   ⏱️  Time:       {batch_time:.2f} s")
    print(f"   ⚡ Throughput: {batch_ips:.0f} items/sec")
    print(f"   🔥 Speedup:    {speedup:.1f}x faster")

    # --- TEST 3: SEQUENTIAL SEARCH LATENCY ---
    print(f"\n[3/4] 🔍 Testing Sequential Search Latency ({SEARCH_ITERATIONS} iters)...")
    # Warmup
    for _ in range(10): db.search("warmup", 5)
    
    latencies = []
    query = "market inflation analysis"
    
    gc.collect()
    start_total = time.perf_counter()
    for _ in range(SEARCH_ITERATIONS):
        t0 = time.perf_counter_ns()
        db.search(query, k=5)
        t1 = time.perf_counter_ns()
        latencies.append((t1 - t0) / 1000.0) # microseconds
    end_total = time.perf_counter()
    
    seq_qps = SEARCH_ITERATIONS / (end_total - start_total)
    
    print(f"   ⚡ QPS:        {seq_qps:.0f} queries/sec")
    print(f"   ⏱️  Avg Latency: {statistics.mean(latencies):.2f} μs")
    print(f"   ⏱️  P50 (Med):   {statistics.median(latencies):.2f} μs")
    print(f"   ⏱️  P99 (Tail):  {statistics.quantiles(latencies, n=100)[98]:.2f} μs")

    # --- TEST 4: CONCURRENT SEARCH THROUGHPUT ---
    print(f"\n[4/4] 🌐 Testing Concurrent Search ({CONCURRENT_THREADS} threads)...")
    # This tests if the Rust Engine correctly releases the GIL and uses the Session Pool
    
    def search_worker(n):
        for _ in range(n):
            db.search("concurrency test", k=5)

    iters_per_thread = SEARCH_ITERATIONS // CONCURRENT_THREADS
    
    gc.collect()
    start_conc = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_THREADS) as executor:
        futures = [executor.submit(search_worker, iters_per_thread) for _ in range(CONCURRENT_THREADS)]
        concurrent.futures.wait(futures)
    end_conc = time.perf_counter()
    
    conc_time = end_conc - start_conc
    conc_qps = SEARCH_ITERATIONS / conc_time
    
    print(f"   ⏱️  Total Time: {conc_time:.2f} s")
    print(f"   ⚡ QPS:        {conc_qps:.0f} queries/sec")
    if conc_qps > seq_qps:
        print(f"   ✅ Scaling:    True (+{((conc_qps/seq_qps)-1)*100:.1f}%)")
    else:
        print(f"   ⚠️ Scaling:    False (Bottlenecked)")

    print("-" * 60)

if __name__ == "__main__":
    run_benchmark()
