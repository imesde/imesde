# ⚡️ imesde

> **"Semantic search at the speed of a pipe. No disk, no lag, just context."**

![Imesde Demo](assets/demo.gif)

Welcome to **imesde**, the **In-Memory Streaming Data Engine**. We are defining a new category in the data landscape: the **IMESD**.

---

## 🎯 Why imesde?

Traditional vector databases are built for persistence and long-term storage. imesde is built for **speed and ephemerality**:

- **Zero-Disk Dependency**: Pure RAM operation. Old data flows out as new data flows in. No GC, no fragmentation. Optimized with **SIMD-accelerated** dot product kernels for maximum CPU efficiency.
- **Lock-Free Architecture**: High-throughput ingestion and search using atomic operations and sharded buffers.
- **Local-First Privacy**: In-process vectorization (ONNX) and storage. Data never leaves your machine.
- **Unix Philosophy**: Designed to be pipe-friendly. `tail -f logs | imesde`.

## 💡 Use Cases (The Power of Ephemeral Context)

**imesde** is designed for scenarios where you need to process high-speed streams and perform semantic analysis on the "now" without the overhead of a database cluster.

### 1. Semantic Market Monitoring
Monitor financial news feeds (RSS, Bloomberg, Twitter) for complex semantic triggers. Instead of keyword matching (which misses context), use **imesde** to detect "inflationary pressure" or "supply chain disruption" across thousands of headlines in real-time.
*See: `bindings/python/examples/market_monitor.py`*

### 2. Live Firehose Filtering (Wikipedia/Social Media)
Connect to a massive event stream (like the Wikipedia Recent Changes firehose) and perform live semantic filtering. Detect specific topics (e.g., "international diplomacy", "tech breakthroughs") as they happen, using the circular buffer to keep only the most recent context.
*See: `bindings/python/examples/wikipedia_event_stream.py`*

### 3. Log Anomaly Detection
Pipe your server logs into **imesde**. Perform periodic searches for "security breach attempts" or "unusual database failures". The circular buffer ensures you always have a sliding window of the last 16k logs available for semantic query, automatically discarding the old ones.

---

## 🏎 Performance (Benchmark Results)

### 📥 Ingestion Throughput (1,000 records)
| Method | Time (s) | Improvement |
| :--- | :--- | :--- |
| **Single Ingest** | 16.29 s | - |
| **Batch Ingest (Parallel)** | **10.60 s** | **+53.7%** |

### 🔍 Search Latency (1,000 queries)
| Metric | Latency (μs) |
| :--- | :--- |
| **Average** | 18,872.74 μs |
| **Minimum** | 17,598.92 μs |
| **P99 (Worst)** | 21,330.84 μs |
| **Maximum** | 54,969.25 μs |

*Benchmarks executed on 1,000 records with a sharded circular buffer and ONNX in-process inference.*

## 🛠 Technical DNA

- **Language**: Rust
- **Engine**: Sharded Lock-Free Circular Buffer
- **Inference**: In-process ONNX Runtime
- **Target**: 10,000+ ingestions/sec on standard hardware

## 🧩 Architecture

imesde isn't a traditional database. It's a high-speed pipeline:

`[Stream Input] -> [Parallel ONNX Embedding] -> [Sharded Circular Buffer] -> [Top-K SIMD Search]`

---

## 🚀 Quick Start
```bash
# Build Rust binary
cargo build --release
```

> **Note on Models**: imesde requires an ONNX model (e.g., `all-MiniLM-L6-v2`) and its `tokenizer.json`. You can export these from Hugging Face using `optimum-cli` or `sentence-transformers`. Place them in the `model/` directory as `model.onnx` and `tokenizer.json`.

## 🐍 Python Usage
You can install **imesde** as a Python module:

```bash
cd bindings/python
maturin build --release
pip install ../../target/wheels/imesde-*.whl
```

### Example
```python
from imesde import PyImesde

# Initialize with model paths
engine = PyImesde("model/model.onnx", "model/tokenizer.json")

# Single ingestion
engine.ingest("Real-time log data flow")

# High-performance batch ingestion (Parallelized)
logs = ["User login at 10:00", "DB Query took 500ms", "Connection reset"]
engine.ingest_batch(logs)

# Search the circular buffer
results = engine.search("database issues", k=5)
for text, score in results:
    print(f"[{score:.4f}] {text}")
```

---
*Note: Requires `model.onnx` and `tokenizer.json` in the `model/` directory.*

---
*MIT Licensed. Built for the speed of thought.*