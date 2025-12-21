![Rust](https://img.shields.io/badge/Rust-000000?style=flat&logo=rust&logoColor=white)
![Python](https://img.shields.io/badge/Python-14354C?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue)

# ⚡️ imesde

> **The Real-Time Vector Database**. "Semantic search at the speed of a pipe. No disk, no lag, just context."

![Imesde Demo](assets/demo.gif)

Welcome to **imesde**, the **In-Memory Streaming Data Engine**. We are defining a new category in the data landscape: the **IMESD**.

---

## 🎯 Why imesde?

Traditional vector databases are built for persistence and long-term storage. imesde is built for **speed and ephemerality**:

- **Zero-Disk Dependency**: Pure RAM operation. Old data flows out as new data flows in. No GC, no fragmentation. Optimized with **SIMD-accelerated** dot product kernels for maximum CPU efficiency.

- **Lock-Free Architecture**: High-throughput ingestion and search using sharded buffers.

- **Local-First Privacy**: In-process vectorization (ONNX) and storage. Data never leaves your machine.

- **Unix Philosophy**: Designed to be pipe-friendly. `tail -f logs | imesde`.

---

### Use Cases

| Use Case | imesde | Traditional Vector DB |
| :--- | :--- | :--- |
| **Live Firehose (Logs/Tweets)** | ✅ **Best** (Circular Buffer) | ❌ Slow (Disk/Indexing lag) |
| **Search 10M PDF Documents** | ❌ No (RAM limited) | ✅ **Best** (Disk/HNSW) |
| **Privacy-First / Edge** | ✅ **Best** (Zero-deps) | ❌ Hard (Heavy services) |
| **Infrastructure Cost** | 💎 **Minimal (Single binary)** | 💸 High (Cloud/Cluster) |

---

## 🚀 imesde Performance Benchmark

💻 **System**: Apple M4 (Darwin 24.6.0) - 16GB RAM  
📊 **Dataset**: 5000 records  
🧠 **Model**: [bge-small-en-v1.5 int8](https://huggingface.co/Xenova/bge-small-en-v1.5/tree/main)

|Metric	|Result|
|---|---|
|Avg Search Latency|232.74 μs|
|P99 Search Latency|383.75 μs|
|Engine Throughput|4,297 queries/sec|
|Avg Embedding Time|1.93 ms|
|Total QPS|662 queries/sec|

### 🏆 Recommended Models

| Model | Format | Best For |
| :--- | :--- | :--- |
| [bge-small-en-v1.5 int8](https://huggingface.co/Xenova/bge-small-en-v1.5/tree/main) | ONNX (Int8) | **Maximum Speed.** Best balance for real-time CPU streams. |
| [all-MiniLM-L6-v2 int8](https://huggingface.co/Xenova/all-MiniLM-L6-v2/tree/main) | ONNX (Int8) | **General Purpose.** Versatile and lightweight. |

> **General Recommendation**: For the best balance of speed and efficiency on CPUs, we generally recommend using **Int8 quantized** models.

---

## 🧠 Why CPU-First?

imesde is intentionally architected to run on **CPUs**, not GPUs.
While GPUs offer high throughput for massive batch training, they introduce **latency** (PCIe data transfer) and operational complexity that contradict the goal of a lightweight, real-time streaming engine.

**The Strategy:**
1. **Zero-Latency**: No data transfer between RAM and VRAM.
2. **Quantization is King**: Modern CPUs with AVX2/NEON/AMX process **Int8 Quantized** models at monstrous speeds.
3. **Result**: GPU-class inference for streaming data with significantly lower operational complexity.


> **Need higher precision?** If absolute semantic accuracy > latency, you can simply drop in a standard Float32 model (e.g., `bge-large`, `e5-mistral`). imesde works with any ONNX model out of the box.

---

## 🚀 Quick Start
```bash
# Build Rust binary
cargo build --release
```

## 🐍 Python Usage
For a detailed guide on using imesde with Python, see the [Python Documentation](docs/python_usage.md).

You can install **imesde** directly via pip:

```bash
pip install imesde
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

> *Note: imesde requires an ONNX model and its tokenizer. You can export these from Hugging Face using `optimum-cli` or `sentence-transformers`. Place them in the `model/` directory as `model.onnx` and `tokenizer.json`.*

---
*MIT Licensed. Built for the speed of thought.*
