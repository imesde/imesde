# 🛰️ Showcase: Semantic Radar & Stress Test

![Semantic Radar Demo](../assets/semantic_radar.gif)

🛰️ **Semantic Radar Demo**: This example demonstrates how to use **imesde** to monitor a global live stream of data from the OpenSky Network and perform real-time semantic analysis to detect aviation anomalies. 

In this demo, imesde processes **~10,000 flight status updates per minute** via the OpenSky API. The system performs local vector embedding and semantic search to identify flight anomalies in **less than 1ms per record**, all running entirely on the CPU.

## 🧠 Key Concept: Semantic Mapping

One of the most powerful ways to use `imesde` is by transforming raw numerical data into **descriptive semantic tags**. 

Instead of searching for a value like `altitude < 1000`, we feed the engine descriptive strings:
- *Raw*: `alt: 300, vel: 450`
- *Semantic*: `"Flight at low altitude and extreme high speed."`

The embedding model understands the **intent** behind "danger" or "emergency" better than a simple database query.

```python
# Mapping telemetry to concepts
alt_tag = "extreme altitude" if alt > 11000 else "low altitude" if alt < 1000 else "standard flight level"
vel_tag = "supersonic/high speed" if vel > 280 else "slow speed" if vel < 100 else "normal cruise speed"
status_tag = "EMERGENCY code detected" if squawk in ["7700", "7600", "7500"] else "routine flight"

# The final semantic string stored in imesde
rich_report = f"Flight {callsign}. Status: {status_tag}. {alt_tag} at {alt}m. Speed: {vel_tag}."
```

## 🚀 Performance at Scale

In this showcase, we ingest the entire global airspace (typically **10,000+ aircraft states**) in a single batch.

### Parallel Sharding
We initialize the engine with high sharding to maximize throughput:
```python
db = imesde.PyImesde(
    "model/model.onnx", 
    "model/tokenizer.json", 
    num_shards=32,   # Parallelize across 32 shards
    shard_size=2048  # Total capacity: 65,536 vectors
)
```

### High-Speed Ingestion & Search
## 🤖 AI Reasoning & Integration (Tiered Gating)

The system uses a **Tiered Alerting** architecture to minimize computational costs. The LLM (Ollama) is only invoked when Rust-based filters detect significant anomalies:

| Tier | Method | Trigger | Purpose |
| :--- | :--- | :--- | :--- |
| **1. Targeted** | `db.search()` | Score > 0.70 | Immediate danger (Known Red Flags like "Emergency"). |
| **2. Systemic** | `Drift Analysis` | Similarity < 0.98 | Global pattern shift (Storms, Groundings, Major events). |
| **3. Statistical**| `db.get_outliers()`| Sim to Mean < 0.45 | Unique anomalies (Unknown "weird" behavior). |

This approach ensures that **99% of the monitoring happens at the Rust level** (sub-millisecond), reserving the expensive AI reasoning only for confirmed points of interest.


This approach is significantly more efficient than feeding every raw event to an LLM, as `imesde` acts as a high-speed semantic filter.

### Example Analysis Output (Flight CXK168)
> **[AI] Reasoning on top match...**
> 
> 🤖 **ANALYSIS:**
> *   **Status**: Flight CXK168 is operating at low altitude and reduced airspeed within a stable airspace sector.
> *   **Reason**: Intentional slow-speed maneuver to mitigate low-level atmospheric turbulence and erratic airflows (vortices) near terrain.
> *   **Risk**: **Low/Manageable**. Airspeed control ensures structural stability; continuous ATC monitoring confirms zero collision risk.
>
> **Summary**: Flight CXK168 is following standard safety procedures for low-altitude unstable air conditions. No immediate danger.

## 🏃 How to Run the Showcase

To see the Semantic Radar in action, follow these steps:

1. **Install Dependencies**:
   ```bash
   pip install requests ollama imesde
   ```
2. **Ensure Ollama is running** with the model loaded:
   ```bash
   ollama run phi3
   ```
3. **Run the Script**:
   ```bash
   python bindings/python/examples/semantic_radar.py
   ```

### 🚀 View Full Showcase
To see the complete, runnable implementation including the API fetching and performance logging, check out the source file:
👉 **[bindings/python/examples/semantic_radar.py](../bindings/python/examples/semantic_radar.py)**

---
*Back to [Python Usage](python_usage.md) or [Real-Time RAG](rag_engine.md).*
