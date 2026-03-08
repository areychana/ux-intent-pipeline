from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

# Sample UX research transcriptions (longer, more realistic)
transcripts = [
    "I don't understand where to click to save my work",
    "The save button is really hard to find",
    "I keep losing my progress because I can't find how to save",
    "This button doesn't do anything when I press it",
    "The export button is broken, nothing happens",
    "I clicked submit three times and nothing happened",
    "Wow this is really easy to use, I love it",
    "The interface is very intuitive and clean",
    "I finished the task very quickly, great design",
    "Why can't I export my results? I need that for my report",
    "I've been trying to find the settings for 10 minutes",
    "Where is the settings page? I can't find it anywhere"
]

# Cluster into topics
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(transcripts)

num_topics = 3
km = KMeans(n_clusters=num_topics, random_state=42, n_init=10)
km.fit(X)

print("=== UX Topic Extractor ===\n")

for topic_id in range(num_topics):
    print(f"Topic {topic_id + 1}:")
    topic_transcripts = [transcripts[i] for i, label in enumerate(km.labels_) if label == topic_id]
    for t in topic_transcripts:
        print(f"  - {t}")
    print()
