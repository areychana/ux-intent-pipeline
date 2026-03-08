from transformers import pipeline

# Load zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# These are the intents we want to detect (UX research specific)
intents = [
    "user is confused",
    "user wants help",
    "user found a bug",
    "user is satisfied",
    "user wants a new feature",
    "user is frustrated"
]

# Sample UX research transcription segments
transcripts = [
    "I don't understand where to click to save my work",
    "This button doesn't do anything when I press it",
    "Wow this is really easy to use, I love it",
    "Why can't I export my results? I need that for my report",
    "I've been trying to find the settings for 10 minutes"
]

print("=== UX Intent Detector ===\n")

for text in transcripts:
    result = classifier(text, candidate_labels=intents)
    top_intent = result["labels"][0]
    confidence = result["scores"][0]
    print(f"Text: {text}")
    print(f"Intent: {top_intent} ({confidence:.0%} confidence)")
    print()