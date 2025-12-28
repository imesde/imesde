import imesde
import time
import numpy as np

def main():
    # 1. Initialize imesde
    # The model and tokenizer should be in the 'model/' directory
    db = imesde.PyImesde("model/model.onnx", "model/tokenizer.json")

    print("🚀 Centroid-based Anomaly Detection Demo")
    
    # 2. Define "Normal" behavior data
    # Let's simulate a series of standard log messages or telemetry
    normal_data = [
        "System heartbeat: OK",
        "Memory usage: 45%",
        "CPU temperature: 52C",
        "Network latency: 12ms",
        "Disk I/O: normal",
        "User session active: id_882",
        "Cache hit ratio: 94%",
        "Database connection pool: 12/50",
        "Background task completed: cleanup_logs",
        "API status: healthy"
    ]

    print(f"📥 Ingesting {len(normal_data)} normal events...")
    db.ingest_batch(normal_data)

    # 3. Calculate the Centroid (the "Mathematical Mean")
    # This represents the 'average' state of the system
    centroid = db.get_centroid()
    if centroid:
        print(f"✅ Mathematical mean (centroid) calculated. Vector size: {len(centroid)}")

    # 4. Ingest an Anomaly
    # Something semantically very different
    anomalies = [
        "CRITICAL ERROR: Kernel panic detected! Memory corruption at 0x442",
        "SECURITY ALERT: Multiple failed login attempts from unknown IP",
        "Hardware failure: Fan stopped working, temperature critical"
    ]
    
    print(f"⚠️ Ingesting {len(anomalies)} anomalous events...")
    db.ingest_batch(anomalies)

    # 5. Detect Outliers
    # Threshold: 1.0 means identical, 0.0 means orthogonal.
    # Anything below the threshold is considered an outlier.
    # For these models, anything below 0.60 - 0.70 from the mean might be suspicious
    # depending on the diversity of the 'normal' data.
    THRESHOLD = 0.55
    
    print(f"🔍 Searching for outliers (Similarity to mean < {THRESHOLD})...")
    
    # Let's see all scores first
    all_scores = db.get_scores_from_centroid()
    print("\n📊 Current distribution (Similarity to Mean):")
    for record, score in sorted(all_scores, key=lambda x: x[1], reverse=True):
        is_anomaly = "⚠️" if score < THRESHOLD else "✅"
        print(f"  {is_anomaly} [SCORE: {score:.4f}] {record}")

    outliers = db.get_outliers(THRESHOLD)

    if outliers:
        print(f"🔥 Found {len(outliers)} outliers moving too far from the mean:")
        for record, score in outliers:
            print(f"  - [SCORE: {score:.4f}] {record}")
    else:
        print("✅ No outliers detected.")

    # 6. Real-time monitoring simulation with INSTANT anomaly detection
    print("\n🔄 Simulating real-time monitoring (Instant O(1) detection)...")
    
    new_events = [
        "Routine database backup started.",
        "UNAUTHORIZED ACCESS: Root password changed via SSH."
    ]

    for event in new_events:
        # Ingest now returns the similarity to the current mean immediately!
        score = db.ingest(event)
        is_anomaly = "🔥 YES" if score < THRESHOLD else "✅ NO"
        print(f"New event: '{event}'")
        print(f"  - Similarity to mean: {score:.4f} -> Outlier? {is_anomaly}")

if __name__ == "__main__":
    main()
