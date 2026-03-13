from src.intent_detector import detect_intent
from src.topic_extractor import extract_topics
from typing import Dict, List
import os
import json


def _llm_summary(text: str, intent: str, topics: List[str]) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _template_summary(intent, topics)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            f"You are a UX researcher. A user gave this feedback:\n\n"
            f"\"{text}\"\n\n"
            f"The detected intent is '{intent}' and the main topics are: {', '.join(topics)}.\n"
            f"Write ONE concise sentence summarizing the UX insight for a research report."
        )
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except Exception:
        return _template_summary(intent, topics)


def _template_summary(intent: str, topics: List[str]) -> str:
    intent_readable = intent.replace("_", " ")
    topic_str = ", ".join(topics[:3]) if topics else "general UX friction"
    return f"User feedback indicates a {intent_readable} related to {topic_str}."


def run_pipeline(text: str) -> Dict:
    if not text or not text.strip():
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "topics": [],
            "summary": "No feedback text provided.",
        }

    intent, confidence = detect_intent(text)

    sentences = [s.strip() for s in text.replace(".", ".\n").split("\n") if s.strip()]

    if len(sentences) >= 2:
        raw_topics = extract_topics(sentences, n_clusters=2, top_keywords=3)
        topics = [kw for t in raw_topics for kw in t["keywords"][:2]]
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        vec = TfidfVectorizer(stop_words="english", max_features=6)
        try:
            X = vec.fit_transform([text])
            topics = list(vec.get_feature_names_out())
        except Exception:
            topics = []

    seen = set()
    unique_topics = []
    for t in topics:
        if t not in seen:
            seen.add(t)
            unique_topics.append(t)

    summary = _llm_summary(text, intent, unique_topics[:4])

    return {
        "intent": intent,
        "confidence": confidence,
        "topics": unique_topics[:6],
        "summary": summary,
    }
