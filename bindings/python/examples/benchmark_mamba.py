import imesde
import time
import statistics
import platform
import gc
import concurrent.futures
import os

# Configuration
# Using the local Mamba-2 weights
MODEL_PATH = "model/mamba2-130m"
TOKENIZER_PATH = "" # Not used directly when pointing to a model folder
NUM_RECORDS = 2       # Mamba is heavier than ONNX on CPU, using smaller dataset
SEARCH_ITERATIONS = 50  # Balanced for Mamba throughput
PURE_SEARCH_ITERATIONS = 500 # Vector scan is fast even with Mamba
CONCURRENT_THREADS = 2  # Mamba is memory/CPU intensive
SHARD_SIZE = 1024
NUM_SHARDS = 8          # Fewer shards needed for small dataset

def get_system_info():
    try:
        return f"{platform.system()} {platform.release()} ({platform.machine()})"
    except:
        return "Unknown System"

def run_benchmark():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model directory {MODEL_PATH} not found.")
        return

    print(f"\n🚀 imesde Performance Benchmark (MAMBA-2 MODE)")
    print(f"💻 System:  {get_system_info()}")
    print(f"📊 Dataset: {NUM_RECORDS} records")
    print(f"📦 Backend: Mamba-2 (State Space Model)")
    print("-" * 60)

    # 1. Initialization
    print(f"🔄 Initializing Mamba Engine...")
    start_init = time.perf_counter()
    db = imesde.PyImesde(MODEL_PATH, TOKENIZER_PATH, num_shards=NUM_SHARDS, shard_size=SHARD_SIZE, use_mamba=True)
    end_init = time.perf_counter()
    print(f"✅ Loaded in {end_init - start_init:.2f} s")

    texts = [f"Aviation telemetry report {i}: altitude stable, engine parameters nominal." for i in range(NUM_RECORDS)]
    keys = [f"ACFT_{i % 10}" for i in range(NUM_RECORDS)]

    # --- TEST 1: STATEFUL BATCH INGESTION ---
    print(f"\n[1/5] 🚀 Ingesting Data (Stateful Propagation)...")
    start_time = time.perf_counter()
    db.ingest_batch(texts, keys=keys)
    end_time = time.perf_counter()
    ingest_latency = end_time - start_time
    print(f"   ⏱️  Total Time:   {ingest_latency:.2f} s")
    print(f"   ⚡ Throughput:   {NUM_RECORDS / ingest_latency:.2f} vectors/sec")

    # --- TEST 2: MAMBA INFERENCE LATENCY ---
    print(f"\n[2/5] 🧠 Measuring Mamba-2 Inference Latency...")
    # Warmup
    _ = db.embed_query("warmup")
    
    embed_latencies = []
    for _ in range(10): 
        t0 = time.perf_counter_ns()
        _ = db.embed_query("dangerous flight maneuver at low altitude detected")
        t1 = time.perf_counter_ns()
        embed_latencies.append((t1 - t0) / 1_000_000.0) # ms
    
    avg_embed = statistics.mean(embed_latencies)
    print(f"   ⏱️  Avg Inference: {avg_embed:.2f} ms")
    print(f"   ℹ️  This includes the full SSM state propagation cost.")

    # --- TEST 3: PURE ENGINE LATENCY (Search Raw) ---
    print(f"\n[3/5] ⚡ Measuring Engine Vector Scan Latency (Pure Search)...")
    query_vec = db.embed_query("emergency squawk code 7700")
    
    search_raw_latencies = []
    for _ in range(PURE_SEARCH_ITERATIONS):
        t0 = time.perf_counter_ns()
        _ = db.search_raw(query_vec, k=5)
        t1 = time.perf_counter_ns()
        search_raw_latencies.append((t1 - t0) / 1000.0) # μs
    
    avg_pure = statistics.mean(search_raw_latencies)
    print(f"   ⏱️  Avg Pure Search: {avg_pure:.2f} μs")
    print(f"   🚀 Engine OPS:      {1_000_000 / avg_pure:.0f} queries/sec")

    # --- TEST 4: HYBRID SEARCH (AI + Vector Scan) ---
    print(f"\n[4/5] ⚡ Measuring Full Search Latency (End-to-End)...")
    
    search_latencies = []
    for _ in range(SEARCH_ITERATIONS):
        t0 = time.perf_counter_ns()
        _ = db.search("emergency squawk code 7700", k=5)
        t1 = time.perf_counter_ns()
        search_latencies.append((t1 - t0) / 1_000_000.0) # ms
    
    avg_search = statistics.mean(search_latencies)
    print(f"   ⏱️  Avg Full Search: {avg_search:.2f} ms")
    print(f"   🚀 System OPS:       {1000 / avg_search:.2f} queries/sec")

    # --- TEST 5: STATEFUL CONTINUITY ---
    print(f"\n[5/5] 📈 Testing State Continuity...")
    
    icao = "TEST_ACFT"
    t1 = "Aircraft stable at 10000m."
    t2 = "Aircraft descending rapidly to 2000m!"
    
    db.ingest(t1, key=icao)
    _ = db.embed_query(t2) # State updated
    
    print(f"   ✅ Successfully processed stateful sequence for {icao}")
    print("-" * 60)

if __name__ == "__main__":
    run_benchmark()