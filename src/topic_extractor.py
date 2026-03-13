from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from typing import List, Dict
import numpy as np


def extract_topics(
    texts: List[str],
    n_clusters: int = None,
    top_keywords: int = 5,
) -> List[Dict]:
    if len(texts) < 2:
        raise ValueError("Need at least 2 texts for topic extraction.")

    if n_clusters is None:
        n_clusters = max(2, int(len(texts) ** 0.5))
    n_clusters = min(n_clusters, len(texts))

    vectorizer = TfidfVectorizer(
         stop_words="english",
         max_features=500,
         ngram_range=(1, 2),
         token_pattern=r"(?u)\b[a-zA-Z]{4,}\b",
  )    
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    km.fit(X)
    labels = km.labels_

    topics = []
    for cluster_idx in range(n_clusters):
        center = km.cluster_centers_[cluster_idx]
        top_indices = np.argsort(center)[::-1][:top_keywords]
        keywords = [feature_names[i] for i in top_indices]

        cluster_texts = [texts[i] for i, lbl in enumerate(labels) if lbl == cluster_idx]

        topics.append({
            "keywords": keywords,
            "texts": cluster_texts,
        })

    return topics
