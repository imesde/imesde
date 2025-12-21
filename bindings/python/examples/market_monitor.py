import imesde
import feedparser
import time
from datetime import datetime

def main():
    # 1. Initialize the Vector DB
    # Ensure model.onnx and tokenizer.json are in the same directory
    db = imesde.PyImesde("model/model.onnx", "model/tokenizer.json")

    # 2. Define the RSS feeds (Reuters Business and Finance are good sources)
    RSS_FEEDS = [
        "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html", # CNBC Earnings
        "https://www.cnbc.com/id/10000664/device/rss/rss.html",  # CNBC Finance
        "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069" # CNBC Investing
    ]

    # 3. Define our Trading "Watchlist" (Semantic Triggers)
    # We look for concepts, not just keywords
    WATCHLIST = [
        {"label": "FED & RATES", "query": "Central bank interest rate decisions and inflation reports"},
        {"label": "TECH EARNINGS", "query": "Quarterly financial results and revenue growth of big tech companies"},
        {"label": "COMMODITIES", "query": "Crude oil supply disruptions and gold price movements"},
        {"label": "SEMICONDUCTORS", "query": "Microchip manufacturing and AI hardware demand"}
    ]

    # To avoid processing the same news twice
    processed_titles = set()

    print(f"🚀 Market Monitor Started at {datetime.now().strftime('%H:%M:%S')}")
    print("Monitoring feeds for semantic matches...\n")

    try:
        while True:
            for url in RSS_FEEDS:
                feed = feedparser.parse(url)
                
                new_items = []
                for entry in feed.entries:
                    if entry.title not in processed_titles:
                        new_items.append(entry.title)
                        processed_titles.add(entry.title)

                if new_items:
                    # Ingest the new batch of headlines
                    print(f"📥 Ingesting {len(new_items)} new headlines...")
                    db.ingest_batch(new_items)

                    # Search for relevant news based on our Watchlist
                    for trigger in WATCHLIST:
                        # Search for the top match for each category
                        results = db.search(trigger["query"], k=1)
                        
                        if results:
                            record, score = results[0]
                            # 0.45 - 0.50 is usually a good threshold for these models
                            if score > 0.1:
                                print(f"--- 🚨 SEMANTIC MATCH FOUND ---")
                                print(f"TAG:      [{trigger['label']}]")
                                print(f"NEWS:     {record}")
                                print(f"SCORE:    {score:.4f}")
                                print(f"TIME:     {datetime.now().strftime('%H:%M:%S')}\n")

            # Wait for 60 seconds before checking for new updates
            time.sleep(60)

    except KeyboardInterrupt:
        print("\nStopping Market Monitor...")

if __name__ == "__main__":
    main()