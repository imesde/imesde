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

#### 1. SHARD_SIZE (The Unit of Work)
**What it is:** The number of elements (slots) contained within a single shard.

**Performance Impact:** This determines the "granularity" of your work. Each shard is processed by a single CPU thread. A smaller size (e.g., 512–1024) ensures that the data fits entirely within the L1/L2 CPU Cache, minimizing slow RAM access. However, if the size is too small, the overhead of managing the thread pool may exceed the time spent on actual calculation.

**When to increase:** Increase this value if your dataset is very large or if your calculation (e.g., cosine_similarity) is extremely fast, requiring larger batches to keep the CPU cores busy.

#### 2. NUM_SHARDS (The Parallelism Factor)
**What it is:** The total number of independent partitions in the buffer.

**Performance Impact:** This controls how many "tasks" are available for Rayon’s work-stealing scheduler. On asymmetric processors like the Apple M4, having more shards than physical cores (e.g., 32 shards for 8 cores) is beneficial. It allows high-performance cores to "steal" and process more shards while slower efficiency cores are still working on their first ones.

**When to increase:** Increase this to improve load balancing across different types of CPU cores (P-cores vs. E-core) or to reduce contention during concurrent insertions.

#### Selection Strategy by Hardware

- **Apple Silicon (M1/M2/M3/M4):**
  - *Recommendation:* **32 shards** (High count to leverage mixed P/E cores).
  - *Reasoning:* The MacOS scheduler moves background tasks to Efficiency cores. A higher shard count ensures Performance cores always have chunks to "steal" and process via Rayon.

- **Standard Desktop (Intel Core i7/i9, AMD Ryzen):**
  - *Recommendation:* **2x the logical thread count** (e.g., 32 shards for a 16-thread CPU).
  - *Reasoning:* Modern x86 CPUs with Hyper-Threading benefit from having more tasks than physical cores to keep the execution pipelines saturated during context switches.

- **Cloud Servers (AWS Graviton, AMD EPYC - 64+ Cores):**
  - *Recommendation:* **64 or 128 shards**.
  - *Reasoning:* Massive parallelism requires massive partitioning. High shard counts reduce contention during concurrent ingestion and fully utilize AVX-512 or NEON units across many cores.

#### Optimal Shard Size

The `shard_size` determines the depth of each partition.
- **Small Datasets (< 5,000 records):** Use **512 or 1024**. This prevents the engine from scanning large amounts of "empty" pre-allocated space and keeps the data inside the L1/L2 cache.
- **Large Datasets / High Capacity:** Use **2048 or 4096**. Larger shards reduce the overhead of merging top-K results from a high number of shards.
- **Ideal Range:** For most streaming use cases, **1024** is the gold standard for performance.

- **General Rule:** Start with **32 shards / 1024 size**. If ingestion feels slow, double the shards. If search feels slow on small datasets, halve the shards.

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

---
*For complete examples, see the `bindings/python/examples` folder in the repository.*