import imesde
import time

try:
    db = imesde.PyImesde("model/model.onnx", "model/tokenizer.json")
    print("🚀 Motore imesde pronto per il trading.")
except Exception as e:
    print(f"❌ Errore nel caricamento: {e}")
    exit()

historical_events = [
    "FED raises interest rates by 50 basis points",
    "Bitcoin network upgrade completed successfully",
    "Nvidia reports record-breaking quarterly earnings",
    "Apple announces new iPhone with satellite connectivity",
    "Oil prices surge due to supply chain disruptions"
]

print("\n📥 Popolamento del database storico...")
for news in historical_events:
    db.ingest(news)
    print(f"Registrato: {news[:40]}...")

live_news = "Central bank announces aggressive rate hikes to fight inflation"
print(f"\n🔔 NEWS FLASH: {live_news}")

start_time = time.perf_counter()
results = db.search(live_news, k=2)
end_time = time.perf_counter()

print(f"🔍 Ricerca completata in {(end_time - start_time)*1000:.3f} ms\n")
print("Eventi passati più correlati:")
for metadata, score in results:
    print(f" - [{score:.4f}] {metadata}")
if results[0][1] > 0.71:
    print("\n⚠️ ALTA CORRELAZIONE RILEVATA: Eseguire strategia 'Interest Rate Hike'...")