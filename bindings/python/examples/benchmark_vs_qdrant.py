import time
import statistics
import platform
import sys
import random

# Requirements: pip install qdrant-client numpy
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    import numpy as np
except ImportError:
    print("❌ Missing dependencies. Please run: pip install qdrant-client numpy")
    sys.exit(1)

try:
    import imesde
except ImportError:
    print("❌ imesde not found. Build it with 'maturin develop'.")
    sys.exit(1)

# --- CONFIGURATION ---
NUM_RECORDS = 5000
DIMENSION = 384
SEARCH_ITERATIONS = 5000
MODEL_PATH = "model/model.onnx"
TOKENIZER_PATH = "model/tokenizer.json"

def get_system_info():
    return f"{platform.system()} {platform.release()} ({platform.machine()})"

def run_pure_engine_benchmark():
    print(f"\n🏆 imesde vs Qdrant: Pure Engine Head-to-Head")
    print(f"💻 System: {get_system_info()}")
    print(f"📊 Data:   {NUM_RECORDS} records, {DIMENSION} dims")
    print("-" * 60)

    # 1. PREPARE DATA
    print("[1/3] 🛠️  Generating random vectors and payloads...")
    raw_vectors = np.random.rand(NUM_RECORDS, DIMENSION).astype(np.float32).tolist()
    raw_texts = [f"Payload data for record {i}" for i in range(NUM_RECORDS)]
    query_vec = np.random.rand(DIMENSION).astype(np.float32).tolist()

    # --- ROUND 1: QDRANT ---
    print(f"\n[2/3] 🔵 Testing Qdrant (In-Memory)...")
    q_client = QdrantClient(":memory:")
    if q_client.collection_exists("bench"):
        q_client.delete_collection("bench")
    q_client.create_collection(collection_name="bench", vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE))

    # Ingestion
    points = [PointStruct(id=i, vector=v, payload={"text": t}) for i, (v, t) in enumerate(zip(raw_vectors, raw_texts))]
    start_ingest = time.perf_counter()
    q_client.upsert("bench", points=points)
    end_ingest = time.perf_counter()
    q_ingest_t = end_ingest - start_ingest

    # Search
    q_latencies = []
    for _ in range(SEARCH_ITERATIONS):
        t0 = time.perf_counter_ns()
        q_client.query_points(collection_name="bench", query=query_vec, limit=5, with_payload=True).points
        t1 = time.perf_counter_ns()
        q_latencies.append((t1 - t0) / 1000.0)
    q_avg = statistics.mean(q_latencies)
    print(f"   ⏱️  Ingestion: {q_ingest_t:.4f} s")
    print(f"   ⏱️  Avg Search: {q_avg:.2f} μs")

    # --- ROUND 2: imesde ---
    print(f"\n[3/3] 🟠 Testing imesde (Rust Engine)...")
    db = imesde.PyImesde(MODEL_PATH, TOKENIZER_PATH, num_shards=16, shard_size=1024)

    # Ingestion (Using optimized batch raw ingestion)
    start_ingest = time.perf_counter()
    db.ingest_batch_raw(raw_vectors, raw_texts)
    end_ingest = time.perf_counter()
    i_ingest_t = end_ingest - start_ingest

    # Search
    i_latencies = []
    for _ in range(SEARCH_ITERATIONS):
        t0 = time.perf_counter_ns()
        db.search_raw(query_vec, k=5)
        t1 = time.perf_counter_ns()
        i_latencies.append((t1 - t0) / 1000.0)
    i_avg = statistics.mean(i_latencies)
    print(f"   ⏱️  Ingestion: {i_ingest_t:.4f} s")
    print(f"   ⏱️  Avg Search: {i_avg:.2f} μs")

    # --- RESULTS ---
    print("\n" + "="*60)
    print(f"   🔹 Qdrant: {q_avg:.2f} μs")
    print(f"   🔸 imesde: {i_avg:.2f} μs")
    print(f"   🚀 Speedup: {q_avg / i_avg:.1f}x")
    print("="*60)

if __name__ == "__main__":
    run_pure_engine_benchmark()