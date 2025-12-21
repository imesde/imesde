# 🛰️ Showcase: Semantic Radar & Stress Test

This example demonstrates how to use **imesde** to monitor a global live stream of data (OpenSky Network) and perform real-time semantic analysis to detect aviation anomalies.

## 🧠 Key Concept: Semantic Mapping

One of the most powerful ways to use `imesde` is by transforming raw numerical data into **descriptive semantic tags**. 

Instead of searching for a value like `altitude < 1000`, we feed the engine descriptive strings:
- *Raw*: `alt: 300, vel: 450`
- *Semantic*: `"Flight at low altitude and extreme high speed."`

The embedding model understands the **intent** behind "danger" or "emergency" better than a simple database query.

## 🚀 Performance at Scale

In this showcase, we ingest the entire global airspace (typically **10,000+ aircraft states**) in a single batch.

### What this tests:
1. **Batch Ingestion Throughput**: How many vectors/sec `imesde` can process on your CPU.
2. **Parallel Sharding**: We initialize the engine with `num_shards=32` to maximize CPU utilization during mass ingestion.
3. **Low Latency Search**: Finding the top-K anomalies across the entire global buffer in milliseconds.

## 🤖 AI Reasoning & Example Output

The system uses **Ollama** (specifically the `phi3` model) to analyze only the most relevant matches. Below is an example of the schematic analysis generated when the radar detects an anomaly:

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

---
*Back to [Python Usage](python_usage.md) or [Real-Time RAG](rag_engine.md).*
