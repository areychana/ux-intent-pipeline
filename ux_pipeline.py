from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import json

# Load intent classifier
print("Loading model...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Intents to detect
intents = [
    "user is confused",
    "user wants help",
    "user found a bug",
    "user is satisfied",
    "user wants a new feature",
    "user is frustrated"
]

# Sample UX research transcriptions
transcripts = [
    "I don't understand where to click to save my work",
    "This button doesn't do anything when I press it",
    "Wow this is really easy to use, I love it",
    "Why can't I export my results? I need that for my report",
    "I've been trying to find the settings for 10 minutes",
    "The save button is really hard to find",
    "I keep losing my progress because I can't find how to save",
    "The interface is very intuitive and clean",
]

# Step 1: Detect intents
print("Detecting intents...\n")
results = []
for text in transcripts:
    result = classifier(text, candidate_labels=intents)
    results.append({
        "text": text,
        "intent": result["labels"][0],
        "confidence": round(result["scores"][0], 2)
    })

# Step 2: Extract topics
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(transcripts)
km = KMeans(n_clusters=3, random_state=42, n_init=10)
km.fit(X)

for i, r in enumerate(results):
    r["topic"] = f"Topic {km.labels_[i] + 1}"

# Step 3: Generate structured report
report = {
    "total_segments": len(transcripts),
    "intent_summary": {},
    "topic_summary": {},
    "details": results
}

for r in results:
    intent = r["intent"]
    topic = r["topic"]
    report["intent_summary"][intent] = report["intent_summary"].get(intent, 0) + 1
    report["topic_summary"][topic] = report["topic_summary"].get(topic, 0) + 1

print("=== UX ANALYSIS REPORT ===\n")
print(f"Total segments analyzed: {report['total_segments']}")
print(f"\nIntent Summary: {json.dumps(report['intent_summary'], indent=2)}")
print(f"\nTopic Summary: {json.dumps(report['topic_summary'], indent=2)}")
print("\nDetailed Results:")
for r in results:
    print(f"  [{r['topic']}] [{r['intent']} - {r['confidence']}] {r['text']}")

# Save report to JSON
with open("ux_report.json", "w") as f:
    json.dump(report, f, indent=2)
print("\nReport saved to ux_report.json")
