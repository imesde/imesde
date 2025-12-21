# 🤖 AI & LLM Integration Guide

This guide demonstrates how to use **imesde** as a real-time context provider for Large Language Models. By combining the speed of `imesde` with the reasoning of LLMs, you can build autonomous systems that react to live data streams.

## 💼 Business Case: The "SRE Sentinel"

**Problem**: Traditional log analysis tools (like ELK) have indexing latency. In a fast-moving security breach (e.g., a brute-force attack), every second counts.
**Solution**: Use `imesde` to ingest a live stream of auth logs and use an LLM to perform "Zero-Day" reasoning on the most recent 1,000 events.

---

## 🏠 Local Integration (with Ollama)

Ideal for **Privacy-First** environments where logs should never leave the server.

```python
import imesde
import requests
import json

# 1. Setup imesde
db = imesde.PyImesde("model/model.onnx", "model/tokenizer.json")

# 2. Simulate ingesting live logs (e.g., from a web server)
logs = [
    "192.168.1.45 - [21/Dec/2025:10:01:02] POST /login HTTP/1.1 401",
    "192.168.1.45 - [21/Dec/2025:10:01:03] POST /login HTTP/1.1 401",
    "192.168.1.45 - [21/Dec/2025:10:01:05] POST /login HTTP/1.1 401",
]
db.ingest_batch(logs)

# 3. Retrieve context for a specific suspicion
query = "repeated failed login attempts from the same IP"
context = db.search(query, k=5)
context_str = "\n".join([text for text, score in context])

# 4. Ask Ollama for an assessment
prompt = f"""
Analyze these recent logs and determine if there is an ongoing brute-force attack.
LOGS:
{context_str}

Summary of threat (High/Medium/Low) and recommended action:"""

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
)

print(response.json()['response'])
```

---

## ☁️ Cloud Integration (with OpenAI)

Best for complex reasoning where high intelligence is required to distinguish between noise and real incidents.

```python
from imesde import PyImesde
from openai import OpenAI

client = OpenAI()
db = PyImesde("model/model.onnx", "model/tokenizer.json")

# Retrieve the 'Infinite Window' context
query = "Anomalous patterns in API traffic"
hits = db.search(query, k=10)
context = "\n".join([h[0] for h in hits])

# Execute RAG
completion = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "You are a Security Analyst monitoring live traffic."},
    {"role": "user", "content": f"Review this context: {context}. Is there any unusual pattern?"}
  ]
)

print(completion.choices[0].message.content)
```

## 🛠 Why imesde for AI?

| Feature | Benefit for AI Agents |
| :-- | :-- |
| **Zero Indexing Latency** | The LLM sees events the millisecond they happen. |
| **Circular Memory** | Prevents "Context Pollution"—no stale data from yesterday. |
| **Local Embeddings** | Low latency; no need to pay for embedding APIs for every log line. |

---
*Next: Learn about the [Infinite Window RAG](rag_engine.md).*
