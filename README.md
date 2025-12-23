![Rust](https://img.shields.io/badge/Rust-000000?style=flat&logo=rust&logoColor=white)
![Python](https://img.shields.io/badge/Python-14354C?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue)

# ⚡️ imesde

> **The Real-Time Vector Database**. "Semantic search at the speed of a pipe. No disk, no lag, just context."

![Imesde Demo](assets/demo.gif)

Welcome to **imesde**, the **In-Memory Streaming Data Engine**. We are defining a new category in the data landscape: the **IMESDE**.

---

## 🎯 Why imesde?

Traditional vector databases are built for persistence and long-term storage. imesde is built for **speed and ephemerality**:

- **Zero-Disk Dependency**: Pure RAM operation. Old data flows out as new data flows in. No GC, no fragmentation. Optimized with **SIMD-accelerated** dot product kernels for maximum CPU efficiency.

- **Lock-Free Architecture**: High-throughput ingestion and search using sharded buffers.

- **Real-Time RAG Engine**: Enables the [**"Infinite Window"**](docs/rag_engine.md). Feed LLMs with live context (logs, tickers, chats) with zero indexing latency and automatic "forgetting" of stale data.

- **Local-First Privacy**: In-process vectorization (ONNX) and storage. Data never leaves your machine.

---

### Use Cases

| Use Case | imesde | Traditional Vector DB |
| :--- | :--- | :--- |
| **Live Firehose (Logs/Tweets)** | ✅ **Best** (Circular Buffer) | ❌ Slow (Disk/Indexing lag) |
| **Real-Time RAG (Live Context)** | ✅ **Best** (Zero lag) | ❌ Hard (Stale data/Indexing) |
| **Search 10M PDF Documents** | ❌ No (RAM limited) | ✅ **Best** (Disk/HNSW) |
| **Privacy-First / Edge** | ✅ **Best** (Zero-deps) | ❌ Hard (Heavy services) |
| **Infrastructure Cost** | 💎 **Minimal (Single binary)** | 💸 High (Cloud/Cluster) |

---

## 🚀 imesde Performance Benchmark

💻 **System**: Apple M4 (Darwin 24.6.0) - 16GB RAM  
📊 **Dataset**: 5000 records  
⚙️ **Config**: 16 Shards × 1024 Record Size  
🧠 **Model**: [bge-small-en-v1.5 int8](https://huggingface.co/Xenova/bge-small-en-v1.5/tree/main)

|Metric	|Result|
|---|---|
|Avg Search Latency|211.32 μs|
|P99 Search Latency|302.50 μs|
|Engine Throughput|4,732 queries/sec|
|Avg Embedding Time|1.77 ms|
|Total QPS|734 queries/sec|

### 🏆 imesde vs Qdrant (Pure Engine)

A pure engine-to-engine comparison (excluding AI embedding time) between **imesde** and **Qdrant (In-Memory mode)**. This benchmark measures the raw speed of the underlying Rust search kernels.

**Test Setup:** 20,000 records, 384 dimensions, Apple M4 (Darwin 24.6.0).  
**Source Code:** [`benchmark_vs_qdrant.py`](bindings/python/examples/benchmark_vs_qdrant.py)

| Engine | Ingestion Time | Avg Search Latency | Speed |
| :--- | :--- | :--- | :--- |
| **Qdrant** (In-Memory) | 2.0567 s | 11.28 ms | 1x |
| **imesde** (Rust Engine) | **0.0608 s** | **1.28 ms** | **8.8x Faster** |

#### ⚖️ When to use which?

| Feature | **imesde** | **Qdrant / Pinecone** |
| :--- | :--- | :--- |
| **Best for...** | **Live Streaming / Short-Term Memory** | **Knowledge Base / Long-Term Storage** |
| **Data Scale** | < 500,000 records (Linear Scan) | > 1,000,000 records (HNSW Index) |
| **Ingestion** | **Instant** (Append-only) | Slower (Index overhead) |
| **Persistence** | Ephemeral (RAM only) | Persistent (Disk/Cloud) |
| **Architecture** | Single Binary (Lightweight) | Service/Cluster (Heavy) |

**🚀 Where imesde wins:**
- **🤖 AI Agent Memory (Short-Term):** Ideal for agents that need to recall "what happened in the last 10 minutes." The dataset is small but highly volatile. `imesde` provides instant speed, whereas a full Vector DB would be an unnecessary waste of resources for such a small volume.
- **🌊 High-Frequency Streams:** Critical for system logs, financial feeds, or IoT sensors (~10,000 events/sec) requiring real-time anomaly detection. `imesde` ingests and searches without lag, avoiding the performance hit caused by constant re-indexing in traditional DBs.

### 🏆 Recommended Models

| Model | Format | Best For |
| :--- | :--- | :--- |
| [bge-small-en-v1.5 int8](https://huggingface.co/Xenova/bge-small-en-v1.5/tree/main) | ONNX (Int8) | **Maximum Speed.** Best balance for real-time CPU streams. |
| [all-MiniLM-L6-v2 int8](https://huggingface.co/Xenova/all-MiniLM-L6-v2/tree/main) | ONNX (Int8) | **General Purpose.** Versatile and lightweight. |
| [bge-base-en-v1.5 int8](https://huggingface.co/Xenova/bge-base-en-v1.5/tree/main) | ONNX (Int8) | **High Accuracy.** Better retrieval quality, moderate CPU load. |
| [bge-large-en-v1.5 int8](https://huggingface.co/Xenova/bge-large-en-v1.5/tree/main) | ONNX (Int8) | **Maximum Precision.** SOTA retrieval, highest latency. |

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

### 📖 Documentation & Use Cases
- [**Real-Time RAG (The Infinite Window)**](docs/rag_engine.md): How to use imesde for live context retrieval.
- [**AI & LLM Integration**](docs/ai_integration.md): Examples with **Ollama** and **OpenAI**.
- [**🛰️ Showcase: Semantic Radar**](docs/showcase_radar.md): A full-scale stress test monitoring global aviation data with AI reasoning.
  ```bash
  python bindings/python/examples/semantic_radar.py
  ```

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
