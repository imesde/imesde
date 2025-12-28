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
import numpy as np

# --- CONFIGURATION & PARAMETERS ---
MODEL_NAME = "phi3"  
OPENSKY_URL = "https://opensky-network.org/api/states/all"
FETCH_INTERVAL = 10
MIN_SCORE = 0.70     # Higher threshold for targeted search (more selective)
MIN_OUTLIER_THRESHOLD = 0.40 # Only significantly unique anomalies
DRIFT_THRESHOLD = 0.98       # If global mean changes by more than 2%

# --- UTILS ---
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# --- IMESDE INITIALIZATION ---
db = imesde.PyImesde(
    "model/model.onnx", 
    "model/tokenizer.json", 
    num_shards=32, 
    shard_size=2048
)

def autonomous_alert(flight_data, total_matches, alert_type="Semantic"):
    """
    Step 4: LLM Reasoning (The 'Expensive' Step).
    Called only when automated Rust filters detect a serious issue.
    """
    print(f"\n[AI] 🧠 ACTIVATING REASONING: {alert_type}")
    
    # System-level prompt designed for schematic output
    prompt = f"""
    [SYSTEM: AVIATION SAFETY ANALYZER]
    ALERT TYPE: {alert_type}
    CONTEXT: {total_matches} anomalies found in the current airspace.

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
    Main loop: Performs tiered detection to minimize LLM power consumption.
    """
    print("🚀 IMESDE Smart Semantic Radar Started...")
    previous_centroid = None
    
    while True:
        try:
            # 1. DATA ACQUISITION
            print(f"\n[{time.strftime('%H:%M:%S')}] Scanning Global Airspace...")
            resp = requests.get(OPENSKY_URL, timeout=15)
            states = resp.json().get('states', [])
            
            if not states:
                continue

            # 2. SEMANTIC MAPPING
            reports = []
            for s in states[:5000]: # Limiting for the demo, but capable of handling 10k+
                callsign = s[1].strip() if s[1] else "N/A"
                origin = s[2]
                alt = s[7] if s[7] else 0
                vel = s[9] if s[9] else 0
                squawk = s[14] if s[14] else "0000"
                
                alt_tag = "extreme altitude" if alt > 11000 else "low altitude" if alt < 1000 else "standard flight level"
                vel_tag = "supersonic/high speed" if vel > 280 else "slow speed" if vel < 100 else "normal cruise speed"
                status_tag = "EMERGENCY detected" if squawk in ["7700", "7600", "7500"] else "routine"
                
                rich_report = f"Flight {callsign} ({origin}). Status: {status_tag}. {alt_tag}. Speed: {vel_tag}."
                reports.append(rich_report)

            # 3. FAST INGESTION (Rust)
            # ingest_batch now returns instant anomaly scores for each record
            scores = db.ingest_batch(reports)
            
            # 4. TIERED DETECTION
            # A: SEARCH FOR RED FLAGS (Very specific)
            search_results = db.search("emergency squawk or airplane crash pattern", k=1)
            
            # B: CALCULATE CURRENT CENTROID & DRIFT (Now O(1))
            current_centroid = db.get_centroid()
            drift_alert = False
            if previous_centroid and current_centroid:
                similarity = cosine_similarity(current_centroid, previous_centroid)
                if similarity < DRIFT_THRESHOLD:
                    drift_alert = True
                print(f"📉 Global Stability: {similarity*100:.2f}%")

            # C: IDENTIFY INSTANT OUTLIERS FROM INGESTION SCORES
            # We find the record with the lowest similarity to the mean
            min_score = min(scores) if scores else 1.0
            outlier_idx = scores.index(min_score) if scores else -1

            # 5. SMART ALERTING (LLM Gating)
            # Tier 1: Immediate Danger (Search Match > MIN_SCORE)
            if search_results and search_results[0][1] > MIN_SCORE:
                print(f"🔥 [CRITICAL] Specific Danger Detected!")
                autonomous_alert(search_results[0][0], 1, "CRITICAL_TARGET")
            
            # Tier 2: Systemic Change (Centroid Drift)
            elif drift_alert:
                print(f"⚠️ [SYSTEMIC] Global Airspace Pattern Shift!")
                autonomous_alert("Global shift in telemetry detected across all flights.", len(reports), "GLOBAL_DRIFT")

            # Tier 3: Statistical Weirdness (Instant Outlier)
            elif min_score < MIN_OUTLIER_THRESHOLD:
                print(f"🔍 [ANOMALY] Instant outlier detected (Score: {min_score:.4f})")
                autonomous_alert(reports[outlier_idx], 1, "INSTANT_OUTLIER")
            
            else:
                print(f"✅ Airspace Stable (Best Score: {min_score:.4f}). No LLM required.")

            previous_centroid = current_centroid

        except Exception as e:
            print(f"⚠️ Loop Exception: {e}")
            
        time.sleep(FETCH_INTERVAL)
            
        time.sleep(FETCH_INTERVAL)

if __name__ == "__main__":
    stress_test_loop()
