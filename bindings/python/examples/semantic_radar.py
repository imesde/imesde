"""
🛰️ Semantic Radar Demo: This example demonstrates how to use imesde to monitor a global live stream 
of data from the OpenSky Network and perform real-time semantic analysis to detect aviation anomalies. 

In this demo, imesde processes ~10,000 flight status updates per minute via the OpenSky API. 
The system performs local vector embedding and semantic search to identify flight anomalies 
in less than 1ms per record, all running entirely on the CPU.
"""

import requests
import imesde
import time
import ollama

# --- CONFIGURATION & PARAMETERS ---
MODEL_NAME = "phi3"  # Fast, local LLM for real-time analysis
OPENSKY_URL = "https://opensky-network.org/api/states/all"
FETCH_INTERVAL = 60  # API rate limit safety
MIN_SCORE = 0.60     # Semantic similarity threshold (0.0 to 1.0)

# --- IMESDE INITIALIZATION ---
# Using sharding to distribute the vector load. 
# num_shards=32 allows high-speed parallel ingestion and search on many-core CPUs.
db = imesde.PyImesde(
    "model/model.onnx", 
    "model/tokenizer.json", 
    num_shards=32, 
    shard_size=2048
)

def autonomous_alert(flight_data, total_matches):
    """
    Step 4: LLM Reasoning.
    Only triggered if imesde finds high-relevance anomalies.
    """
    print(f"\n[AI] Reasoning on top match...")

    # System-level prompt designed for schematic output
    prompt = f"""
    [SYSTEM: AVIATION SAFETY ANALYZER]
    CONTEXT: {total_matches} high-relevance semantic matches found in the current airspace.

    DATA SOURCE: 
    {flight_data}

    TASK: Evaluate safety and summarize.
    OUTPUT REQUIREMENTS:
    - Language: English
    - Format: 3 clean bullet points (Status, Reason, Risk)
    - Tone: Schematic, short and professional.
    """

    try:
        response = ollama.generate(model=MODEL_NAME, prompt=prompt)
        print(f"🤖 ANALYSIS:\n{response['response']}\n" + "-"*40)
    except Exception as e:
        print(f"[ERROR] Ollama generation failed: {e}")

def stress_test_loop():
    """
    Main loop: Ingests thousands of flights without limits to test imesde throughput.
    """
    print("🚀 IMESDE Semantic Radar Started...")
    
    while True:
        try:
            # 1. DATA ACQUISITION
            # Fetching the entire global airspace (~10k+ flights)
            print(f"\n[{time.strftime('%H:%M:%S')}] Fetching global aircraft states...")
            resp = requests.get(OPENSKY_URL, timeout=15)
            states = resp.json().get('states', [])
            
            if not states:
                continue

            # 2. SEMANTIC MAPPING (The 'How-To' of imesde)
            # Embeddings work better on text than raw numbers.
            # We map numerical telemetry to descriptive semantic concepts.
            reports = []
            for s in states:
                callsign = s[1].strip() if s[1] else "N/A"
                origin = s[2]
                alt = s[7] if s[7] else 0
                vel = s[9] if s[9] else 0
                squawk = s[14] if s[14] else "0000"
                
                # Semantic labels that the embedding model understands better than "12000m"
                alt_tag = "extreme altitude" if alt > 11000 else "low altitude" if alt < 1000 else "standard flight level"
                vel_tag = "supersonic/high speed" if vel > 280 else "slow speed" if vel < 100 else "normal cruise speed"
                status_tag = "EMERGENCY code detected" if squawk in ["7700", "7600", "7500"] else "routine flight"
                
                # Building the semantic string
                rich_report = f"Flight {callsign} ({origin}). Status: {status_tag}. {alt_tag} at {alt}m. Speed: {vel_tag}."
                reports.append(rich_report)

            # 3. BENCHMARKING INGESTION
            # ingest_batch() is highly optimized for mass ingestion.
            # This is the 'Stress Test' part.
            start_ingest = time.perf_counter()
            db.ingest_batch(reports)
            end_ingest = time.perf_counter()
            
            # 4. SEMANTIC SEARCH
            # We search for a concept ("danger"), not a specific value.
            # imesde calculates the distance between the query and all ingested flights.
            start_search = time.perf_counter()
            search_query = "dangerous high speed at very low altitude or emergency squawk"
            results = db.search(search_query, k=5)
            end_search = time.perf_counter()

            # --- LOGGING STATISTICS ---
            ingest_latency = end_ingest - start_ingest
            search_latency = end_search - start_search
            throughput = len(reports) / ingest_latency if ingest_latency > 0 else 0

            print(f"📊 PERFORMANCE LOG:")
            print(f"  - Vectors Ingested: {len(reports)}")
            print(f"  - Ingestion Latency: {ingest_latency:.4f}s ({int(throughput)} vectors/sec)")
            print(f"  - Search Latency: {search_latency:.4f}s")
            
            # 5. ALERTING
            # Filter results by similarity score. 
            # If score > MIN_SCORE, it's a semantic match for "danger".
            matches = [r for r in results if r[1] > MIN_SCORE]
            if matches:
                print(f"🔥 Found {len(matches)} potential anomalies!")
                autonomous_alert(matches[0][0], len(matches))
            else:
                print("✅ Airspace Scan: No anomalies detected.")

        except Exception as e:
            print(f"⚠️ Loop Exception: {e}")
            
        time.sleep(FETCH_INTERVAL)

if __name__ == "__main__":
    stress_test_loop()
