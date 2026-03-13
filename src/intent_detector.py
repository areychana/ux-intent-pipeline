from transformers import pipeline
from typing import Tuple

_classifier = None

INTENT_LABELS = [
    "navigation issue",
    "feature request",
    "usability complaint",
    "performance issue",
    "positive feedback",
    "confusion or unclear UI",
    "accessibility concern",
    "content issue",
]


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    return _classifier


def detect_intent(text: str) -> Tuple[str, float]:
    if not text or not text.strip():
        return ("unknown", 0.0)

    classifier = _get_classifier()
    result = classifier(text, candidate_labels=INTENT_LABELS)

    top_label = result["labels"][0]
    top_score = round(result["scores"][0], 4)

    normalized = top_label.replace(" ", "_")
    return (normalized, top_score)
