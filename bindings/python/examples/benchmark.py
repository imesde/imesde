import imesde
import time
import statistics

def run_benchmark():
    # 1. Setup: Initialization
    print("--- 🏁 imesde Benchmark Started ---")
    db = imesde.PyImesde("model/model.onnx", "model/tokenizer.json")
    
    num_records = 1000
    texts = [f"Historical financial data point number {i} regarding the stock market" for i in range(num_records)]

    # 2. Population (Standard: One by One)
    print(f"📥 Loading {num_records} records via SINGLE ingestion...")
    start_single = time.perf_counter()
    for t in texts:
        db.ingest(t)
    end_single = time.perf_counter()
    time_single = end_single - start_single
    print(f"✅ Single setup completed in {time_single:.2f} seconds.")

    # 3. Population (New: Parallel Batch)
    # Re-initialize or clear if needed, here we just measure the speed of ingestion
    print(f"📥 Loading {num_records} records via BATCH ingestion (Parallel)...")
    start_batch = time.perf_counter()
    db.ingest_batch(texts)
    end_batch = time.perf_counter()
    time_batch = end_batch - start_batch
    print(f"✅ Batch setup completed in {time_batch:.2f} seconds.")

    # 4. Warm-up phase
    for _ in range(10):
        db.search("test query", k=5)

    # 5. Search measurement (1,000 iterations)
    num_tests = 1000
    latencies_ns = []

    print(f"🔍 Executing {num_tests} semantic searches...")
    for _ in range(num_tests):
        start_ns = time.perf_counter_ns()
        db.search("How is the tech sector performing today?", k=5)
        end_ns = time.perf_counter_ns()
        latencies_ns.append(end_ns - start_ns)

    # 6. Statistical Analysis
    avg_us = (sum(latencies_ns) / num_tests) / 1000
    min_us = min(latencies_ns) / 1000
    max_us = max(latencies_ns) / 1000
    p99_us = statistics.quantiles(latencies_ns, n=100)[98] / 1000

    print("\n--- 📊 SETUP COMPARISON (Ingestion) ---")
    print(f"Single Ingest: {time_single:>8.2f} s")
    print(f"Batch Ingest:  {time_batch:>8.2f} s")
    print(f"Improvement:   {((time_single/time_batch)-1)*100:>8.1f}% faster")

    print("\n--- 📊 SEARCH LATENCY (Microseconds - μs) ---")
    print(f"Average:     {avg_us:>8.2f} μs")
    print(f"Minimum:     {min_us:>8.2f} μs")
    print(f"Maximum:     {max_us:>8.2f} μs")
    print(f"P99 (Worst): {p99_us:>8.2f} μs")
    print("----------------------------------------------")

if __name__ == "__main__":
    run_benchmark()