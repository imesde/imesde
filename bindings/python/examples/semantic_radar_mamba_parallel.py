"""
🛰️  Massive Parallel Mamba Radar: Tracking Global Airspace.
This script demonstrates the full power of imesde:
1. Multi-State Mamba: Each aircraft has its own independent context memory.
2. Rayon Parallelism: Thousands of aircraft updates are processed in parallel.
3. Unified Interface: Everything happens inside db.ingest_batch().
"""

import requests
import imesde
import time

OPENSKY_URL = "https://opensky-network.org/api/states/all"

def main():
    print("🚀 Initializing Massive Stateful Radar...")
    # use_mamba=True activates the multi-state Mamba engine in Rust
    db = imesde.PyImesde("", "", use_mamba=True)
    
    print("📡 Fetching global airspace data...")
    
    # We will do a few iterations to let Mamba build "context" for each flight
    for iteration in range(5):
        try:
            resp = requests.get(OPENSKY_URL, timeout=10)
            states = resp.json().get('states', [])
            if not states: continue
            
            reports = []
            keys = []
            
            # Process up to 2000 flights for this demo to keep it snappy
            for s in states[:2000]:
                icao = s[0]
                callsign = s[1].strip() if s[1] else icao
                alt = s[7] if s[7] else 0
                vel = s[9] if s[9] else 0
                
                # Semantic description
                report = f"Flight {callsign} at {alt}m altitude, speed {vel} m/s."
                
                # INJECT ANOMALY for a specific flight in later iterations
                if iteration == 4 and i == 0:
                    report = f"Flight {callsign} WARNING: UNEXPECTED RAPID DESCENT to 0m!"
                
                reports.append(report)
                keys.append(icao) # The unique ID for the Mamba state

            # --- MASS PARALLEL INGESTION ---
            # This calls Rust + Rayon + Multi-State Mamba
            start = time.perf_counter()
            db.ingest_batch(reports, keys=keys)
            end = time.perf_counter()
            
            print(f"[{iteration}] Processed {len(reports)} flight updates in {end-start:.4f}s")
            
            # Search for anything that sounds like a crash
            # Mamba should give a higher score to the anomaly because it breaks the flight's history
            results = db.search("emergency crash descending rapidly", k=3)
            
            if results and results[0][1] > 0.4:
                 print(f"🔥 Potential Anomaly Detected: {results[0][0]} (Score: {results[0][1]:.4f})")
            else:
                 print("✅ Airspace stable.")
            
            time.sleep(2) # Wait for next update

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
