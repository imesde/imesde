import imesde
import json
import requests
import time

# 1. Initialize DB
db = imesde.PyImesde("model/model.onnx", "model/tokenizer.json")
url = 'https://stream.wikimedia.org/v2/stream/recentchange'

# --- CONFIGURATION ---
MY_QUERY = "Global conflicts, military actions and international diplomacy"
SCORE_THRESHOLD = 0.40 # Only show results that make sense
# ---------------------

headers = {
    'User-Agent': 'ImesdeTestBot/1.0 (contact: alessio@example.com)'
}

print(f"🚀 Connecting to Wikipedia Firehose...")
print(f"🔍 Monitoring for: '{MY_QUERY}'")
print("-" * 50)

buffer = []
batch_size = 256
max_wait_seconds = 5 
last_ingest_time = time.time()

try:
    with requests.get(url, stream=True, headers=headers) as response:
        response.raise_for_status()
        
        for line in response.iter_lines(decode_unicode=True):
            current_time = time.time()
            
            if line and line.startswith("data:"):
                try:
                    json_str = line[5:].strip()
                    data = json.loads(json_str)
                    
                    if data.get('namespace') == 0:
                        title = data.get('title', 'Unknown')
                        comment = data.get('comment', '')
                        if comment:
                            buffer.append(f"{title}: {comment}")
                            
                except json.JSONDecodeError:
                    continue

            # --- INGESTION & LIVE SEARCH LOGIC ---
            time_since_last = current_time - last_ingest_time
            
            if len(buffer) > 0:
                if len(buffer) >= batch_size or time_since_last >= max_wait_seconds:
                    trigger = "SIZE" if len(buffer) >= batch_size else "TIME"
                    
                    # 1. Ingest the data
                    db.ingest_batch(buffer)
                    print(f"📥 Ingested {len(buffer)} edits ({trigger})")

                    # 2. Immediately search the database for your query
                    # This searches EVERYTHING currently in the circular buffer
                    results = db.search(MY_QUERY, k=3)
                    
                    if results:
                        found_any = False
                        for record, score in results:
                            if score > SCORE_THRESHOLD:
                                if not found_any:
                                    print(f"   🎯 Top matches for your query:")
                                    found_any = True
                                print(f"   [Score {score:.4f}] {record}")
                        
                        if not found_any:
                            print("   (No high-confidence matches in this cycle)")
                    
                    print("-" * 50)
                    
                    buffer = []
                    last_ingest_time = current_time
            
except KeyboardInterrupt:
    print("\nStopping stress test...")
except Exception as e:
    print(f"❌ Error: {e}")