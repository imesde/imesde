# 🐍 Python User Guide for imesde

Welcome to the official guide for integrating **imesde** into your Python projects. `imesde` is an in-memory semantic search engine designed for real-time data streams.

## 📦 Quick Installation

To get started, simply install the package via pip:

```bash
pip install imesde
```

## 🚀 Quick Start

Using `imesde` involves three main phases: initialization, data ingestion, and searching.

### 1. Initialization
To start the engine, you need to provide the paths to your ONNX embedding model and its tokenizer. You can also optionally configure the internal circular buffer size.

```python
from imesde import PyImesde

# Initialize with default settings (16 shards x 1024 = 16,384 vectors)
engine = PyImesde("model/model.onnx", "model/tokenizer.json")

# Custom buffer size (e.g., 32 shards x 2048 = 65,536 vectors)
engine = PyImesde(
    "model/model.onnx", 
    "model/tokenizer.json", 
    num_shards=32, 
    shard_size=2048
)
```

> **Note**: `imesde` uses a sharded circular buffer. Total capacity = `num_shards` * `shard_size`.

### 🔧 Advanced Configuration

#### 1. `SHARD_SIZE` (The Unit of Work)
**What it is:** The number of elements (slots) contained within a single shard.

**Performance Impact:** This determines the "granularity" of your work. Each shard is processed by a single CPU thread. A smaller size (e.g., 512–1024) ensures that the data fits entirely within the L1/L2 CPU Cache, minimizing slow RAM access. However, if the size is too small, the overhead of managing the thread pool may exceed the time spent on actual calculation.

**When to increase:** Increase this value if your dataset is very large or if your calculation (e.g., cosine_similarity) is extremely fast, requiring larger batches to keep the CPU cores busy.

#### 2. `NUM_SHARDS` (The Parallelism Factor)
**What it is:** The total number of independent partitions in the buffer.

**Performance Impact:** This controls how many "tasks" are available for Rayon’s work-stealing scheduler. On asymmetric processors like the Apple M4, having more shards than physical cores (e.g., 32 shards for 8 cores) is beneficial. It allows high-performance cores to "steal" and process more shards while slower efficiency cores are still working on their first ones.

**When to increase:** Increase this to improve load balancing across different types of CPU cores (P-cores vs. E-core) or to reduce contention during concurrent insertions.

#### Selection Strategy by Hardware

| Architecture | `num_shards` Recommendation | Reasoning |
| :--- | :--- | :--- |
| **Apple Silicon** (M1/M2/M3/M4) | **32 shards** | Leverages hybrid architecture; P-cores can "steal" tasks from E-cores via Rayon. |
| **Standard Desktop** (i7/i9, Ryzen) | **2x logical threads** | Keeps execution pipelines saturated during context switches (Hyper-Threading). |
| **Cloud Servers** (64+ Cores) | **64 or 128 shards** | Reduces contention during massive ingestion and saturates AVX-512/NEON units. |

#### Optimal Shard Size

| Dataset / Scenario | `shard_size` Recommendation | Benefit |
| :--- | :--- | :--- |
| **Small Datasets** (< 5k records) | **512 or 1024** | Minimizes empty space scanning and keeps data within L1/L2 caches. |
| **Large Datasets** / High Capacity | **2048 or 4096** | Reduces overhead of merging Top-K results across a high number of shards. |
| **Standard Streaming** | **1024** | The "gold standard" for balancing performance and memory locality. |

> **General Rule:** Start with **32 shards / 1024 size**. If ingestion is slow, double the shards. If search is slow on small datasets, halve them.

### 2. Data Ingestion
As a **Circular Buffer**, `imesde` only keeps the most recent data in memory. When the buffer is full, the oldest data is automatically overwritten.

```python
# Single ingestion
db.ingest("New system log detected at 10:30")

# Batch ingestion (Recommended for high performance)
logs = [
    "Database connection error",
    "User 'admin' logged in",
    "Network latency above 200ms"
]
db.ingest_batch(logs)
```

### 3. Semantic Search
You can query the buffer at any time to find the most relevant content relative to a query.

```python
# Search for the top 3 most similar results
results = db.search("network issues", k=3)

for text, score in results:
    print(f"[{score:.4f}] {text}")
```

## 🛠 Model Preparation

`imesde` is model-agnostic, but the files must be provided locally. 

> **General Recommendation**: For the best balance of speed and efficiency on CPUs, we generally recommend using **Int8 quantized** models. If absolute semantic accuracy is more important than latency, you can use standard **Float32** models.

### Recommended Models

| Model | Format | Best For |
| :--- | :--- | :--- |
| [bge-small-en-v1.5 int8](https://huggingface.co/Xenova/bge-small-en-v1.5/tree/main) | ONNX (Int8) | **Production.** Ultra-low latency on CPUs. |
| [all-MiniLM-L6-v2 int8](https://huggingface.co/Xenova/all-MiniLM-L6-v2/tree/main) | ONNX (Int8) | **General Purpose.** Versatile and lightweight. |
| [bge-base-en-v1.5 int8](https://huggingface.co/Xenova/bge-base-en-v1.5/tree/main) | ONNX (Int8) | **High Accuracy.** Better retrieval, moderate speed. |
| [bge-large-en-v1.5 int8](https://huggingface.co/Xenova/bge-large-en-v1.5/tree/main) | ONNX (Int8) | **SOTA Precision.** Best for complex reasoning tasks. |



You can download optimized models from Hugging Face.

 The required files in the `model/` directory are:
- `model.onnx`: The model weights.
- `tokenizer.json`: The file for text tokenization.

## 💡 Common Use Cases

### Real-Time Monitoring
Use `imesde` to analyze tweets, server logs, or RSS feeds as they flow. Instead of searching for exact keywords, you can search for **concepts** (e.g., "security threats" or "market opportunities").

### Context for AI Agents
Use `imesde` as a short-term memory for your LLM agents, providing only the most relevant context retrieved from the recent data stream.

## 🛠 Advanced / Low-Level API

`imesde` exposes low-level methods to bypass the standard pipeline. These are useful for benchmarking, caching, or scenarios where you already have pre-computed vectors.

### 1. `embed_query(text: str) -> List[float]`
Generates the vector embedding for a given text without storing it or searching. Use this to measure the AI model latency or to cache vectors externally.

```python
# Measure how long the AI takes to "understand" a sentence
import time

start = time.perf_counter()
vector = db.embed_query("What is the latency of this model?")
latency_ms = (time.perf_counter() - start) * 1000

print(f"Embedding Latency: {latency_ms:.2f} ms")
# vector is now a list of floats, e.g., [0.12, -0.05, 0.88, ...]
```

### 2. `search_raw(query_vector: List[float], k: int) -> List[Tuple[str, float]]`
Performs a nearest-neighbor search using a raw vector, bypassing the embedding step. This allows for extremely high-frequency searches if the query vector is pre-calculated.

```python
# 1. Pre-calculate the vector (Expensive operation, done once)
query_vec = db.embed_query("critical failure")

# 2. Execute high-frequency search (Cheap operation, done repeatedly)
# This hits the Rust engine directly (sub-millisecond speed)
for _ in range(1000):
    results = db.search_raw(query_vec, k=5)
```

---
*For complete examples, see the `bindings/python/examples` folder in the repository.*