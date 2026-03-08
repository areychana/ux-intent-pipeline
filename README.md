# UX Intent Detection Pipeline

A proof-of-concept NLP pipeline that analyzes UX research transcriptions 
and extracts user intent, recurring themes, and generates structured reports.

Built as part of preparation for GSoC 2026 - Uramaki Lab.

## What it does
- Detects user intent from transcription segments (confused, frustrated, satisfied, etc.)
- Clusters transcriptions into recurring usability themes
- Generates a structured JSON report with intent and topic summaries

## How to run

Install dependencies:
pip install transformers torch scikit-learn

Run the full pipeline:
python ux_pipeline.py

## Sample Output
See sample_output.json for an example report.

## Tech Stack
- Python
- HuggingFace Transformers (facebook/bart-large-mnli)
- scikit-learn (KMeans clustering, TF-IDF)