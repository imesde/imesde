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
    shard_size=2048,
    track_centroid=True # Enable O(1) sliding window anomaly detection (Default: True)
)
```

> **Note**: `imesde` uses a sharded circular buffer. Total capacity = `num_shards` * `shard_size`. Set `track_centroid=False` if you don't need statistical anomaly detection and want maximum ingestion speed.

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
| **Apple Silicon** (M1-M4) | **8-16 shards** (Small Data)<br>**32 shards** (High Load) | For small datasets (<10k), lower overhead wins. For high load, 32 shards engage efficiency cores better. |
| **Standard Desktop** (i7/i9, Ryzen) | **2x logical threads** | Keeps execution pipelines saturated during context switches (Hyper-Threading). |
| **Cloud Servers** (64+ Cores) | **64 or 128 shards** | Reduces contention during massive ingestion and saturates AVX-512/NEON units. |

#### Optimal Shard Size

| Dataset / Scenario | `shard_size` Recommendation | Benefit |
| :--- | :--- | :--- |
| **Small Datasets** (< 5k records) | **512 or 1024** | Minimizes empty space scanning and keeps data within L1/L2 caches. |
| **Large Datasets** / High Capacity | **2048 or 4096** | Reduces overhead of merging Top-K results across a high number of shards. |
| **Standard Streaming** | **1024** | The "gold standard" for balancing performance and memory locality. |

#### Benchmark Comparison (Apple M4)
*Fixed Shard Size: 1024*

| Configuration | Engine OPS | Total QPS (Real World) |
| :--- | :--- | :--- |
| **8 Shards** | 4,241 | 724 |
| **16 Shards** | **4,732** | **734 (Optimal)** |
| **32 Shards** | 4,382 | 714 |
| **64 Shards** | 3,920 | 662 |

> **General Rule:** Start with **16 shards / 1024 size**. If ingestion is slow, double the shards. If search is slow on small datasets, halve them.

### 2. Data Ingestion
As a **Circular Buffer**, `imesde` only keeps the most recent data in memory. When the buffer is full, the oldest data is automatically overwritten.

**New**: `ingest` now returns the **Instant Anomaly Score** ($O(1)$) — the cosine similarity of the new record compared to the current global mean *before* the insertion.

```python
# Single ingestion returns a similarity score (0.0 to 1.0)
score = db.ingest("New system log detected at 10:30")
if score < 0.50:
    print(f"⚠️ Instant anomaly detected! Score: {score}")

# Batch ingestion returns a list of scores
logs = [
    "Database connection error",
    "User 'admin' logged in",
    "Network latency above 200ms"
]
scores = db.ingest_batch(logs)
```

### 3. Semantic Search
You can query the buffer at any time to find the most relevant content relative to a query.

```python
# Search for the top 3 most similar results
results = db.search("network issues", k=3)

for text, score in results:
    print(f"[{score:.4f}] {text}")
```

## 🧠 Centroid-Based Anomaly Detection

In addition to searching with manual queries, `imesde` can automatically identify statistical outliers based on the "mathematical mean" of the current buffer. This is highly effective for detecting anomalies in streams without knowing what you are looking for.

### 1. `get_centroid() -> List[float]`
Calculates the average vector of all records currently in the buffer. This represents the "semantic baseline" of your data stream.

```python
centroid = db.get_centroid()
# centroid is the average concept of everything currently in RAM.
```

### 2. `get_outliers(threshold: float) -> List[Tuple[str, float]]`
Returns all records whose similarity to the mean (centroid) is **lower** than the specified threshold.
- **Threshold 1.0**: Only identical vectors pass.
- **Threshold 0.0**: Everything passes.
- **Typical use**: 0.45 - 0.60 depending on the model and data diversity.

```python
# Find everything that is "weird" compared to the current global state
anomalies = db.get_outliers(threshold=0.55)

for text, score in anomalies:
    print(f"🚨 Statistical Anomaly: {text} (Similarity to mean: {score:.4f})")
```

### 3. `get_scores_from_centroid() -> List[Tuple[str, float]]`
Returns the similarity score to the mean for **every** record in the buffer. Use this to create distribution plots or to dynamically tune your thresholds.

```python
all_scores = db.get_scores_from_centroid()
# Sort to find the most "normal" or most "unique" items
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

### 3. `ingest_raw(vector: List[float], text: str) -> float`
Injects a pre-computed vector directly into the buffer. Returns the instant anomaly score.

```python
vector = [0.1, 0.2, 0.3, ...] # Must match model dimension
score = db.ingest_raw(vector, "My metadata text")
```

### 4. `ingest_batch_raw(vectors: List[List[float]], texts: List[str]) -> List[float]`
High-speed batch ingestion of raw vectors. Bypasses Python loop overhead by processing the entire batch in Rust. Returns a list of instant anomaly scores.

```python
vectors = [[...], [...], [...]]
texts = ["text 1", "text 2", "text 3"]
scores = db.ingest_batch_raw(vectors, texts)
```

---
*For complete examples, see the `bindings/python/examples` folder in the repository.*