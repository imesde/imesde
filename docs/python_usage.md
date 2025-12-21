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
`imesde` requires an embedding model in ONNX format and its corresponding tokenizer.

```python
import imesde

# Initialize the engine
db = imesde.PyImesde("model/model.onnx", "model/tokenizer.json")
```

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