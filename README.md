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

---

### Use Cases

| Use Case | imesde | Traditional Vector DB |
| :--- | :--- | :--- |
| **Live Firehose (Logs/Tweets)** | ✅ **Best** (Circular Buffer) | ❌ Slow (Disk/Indexing lag) |
| **Search 10M PDF Documents** | ❌ No (RAM limited) | ✅ **Best** (Disk/HNSW) |
| **Privacy-First / Edge** | ✅ **Best** (Zero-deps) | ❌ Hard (Heavy services) |

---

## 🚀 imesde Performance Benchmark

💻 **System**: Darwin 24.6.0 (arm64) - 16GB RAM  
📊 **Dataset**: 5000 records  
🧠 **Model**: [bge-small-en-v1.5 int8](https://huggingface.co/Xenova/bge-small-en-v1.5/tree/main)

------------------------------------------------------------

### 🚀 Data Ingestion Latency
⏱️ **Time**: 3.79 s

### 🧠 AI Embedding Latency (CPU/ONNX)
⏱️ **Avg Embedding**: 1928.21 μs (1.93 ms)

### ⚡ Engine Search Latency (Vector Search)
⏱️ **Avg Search**: 232.74 μs  
⏱️ **P99 Search**: 383.75 μs  
🚀 **Engine OPS**: 4297 queries/sec

### 🌐 Concurrent End-to-End Search
⚡ **Total QPS**: 662 queries/sec

---

## 🧠 Why CPU-First?

imesde is intentionally architected to run on **CPUs**, not GPUs.
While GPUs offer high throughput for massive batch training, they introduce **latency** (PCIe data transfer) and complexity (drivers, VRAM management) that contradict the goal of a lightweight, real-time streaming engine.

**The Strategy:**
1.  **Zero-Latency**: By staying on the CPU, we eliminate the overhead of moving data between RAM and VRAM.
2.  **Quantization is King**: Modern CPUs with AVX2/NEON/AMX instructions process **Int8 Quantized** models at monstrous speeds.

*Result*: We achieve GPU-class inference throughput for streaming data with significantly lower latency and operational simplicity.

> **Need higher precision?** If absolute semantic accuracy > latency, you can simply drop in a standard Float32 model (e.g., `bge-large`, `e5-mistral`). imesde works with any ONNX model out of the box.

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
