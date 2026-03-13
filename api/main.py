from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time

from src.intent_detector import detect_intent
from src.topic_extractor import extract_topics
from src.ux_pipeline import run_pipeline

app = FastAPI(
    title="UX Intent Analysis API",
    description="Analyzes UX feedback to detect user intent and extract themes using NLP.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeedbackItem(BaseModel):
    text: str
    source: Optional[str] = "unknown"

class BatchRequest(BaseModel):
    items: List[FeedbackItem]

class IntentResult(BaseModel):
    text: str
    intent: str
    confidence: float
    source: str

class TopicResult(BaseModel):
    topic_id: int
    keywords: List[str]
    representative_texts: List[str]

class PipelineResult(BaseModel):
    intent: str
    confidence: float
    topics: List[str]
    summary: str
    processing_time_ms: float


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/analyze", response_model=PipelineResult)
def analyze_single(item: FeedbackItem):
    start = time.time()
    try:
        result = run_pipeline(item.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = round((time.time() - start) * 1000, 2)
    return PipelineResult(
        intent=result.get("intent", "unknown"),
        confidence=result.get("confidence", 0.0),
        topics=result.get("topics", []),
        summary=result.get("summary", ""),
        processing_time_ms=elapsed,
    )


@app.post("/intent", response_model=IntentResult)
def analyze_intent(item: FeedbackItem):
    try:
        intent, confidence = detect_intent(item.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return IntentResult(
        text=item.text,
        intent=intent,
        confidence=confidence,
        source=item.source,
    )


@app.post("/topics", response_model=List[TopicResult])
def analyze_topics(request: BatchRequest):
    if len(request.items) < 2:
        raise HTTPException(
            status_code=422,
            detail="Topic extraction requires at least 2 texts."
        )
    texts = [item.text for item in request.items]
    try:
        topics = extract_topics(texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return [
        TopicResult(
            topic_id=i,
            keywords=t.get("keywords", []),
            representative_texts=t.get("texts", []),
        )
        for i, t in enumerate(topics)
    ]


@app.post("/batch", response_model=List[PipelineResult])
def analyze_batch(request: BatchRequest):
    results = []
    for item in request.items:
        start = time.time()
        try:
            result = run_pipeline(item.text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error on '{item.text[:40]}': {e}")
        elapsed = round((time.time() - start) * 1000, 2)
        results.append(PipelineResult(
            intent=result.get("intent", "unknown"),
            confidence=result.get("confidence", 0.0),
            topics=result.get("topics", []),
            summary=result.get("summary", ""),
            processing_time_ms=elapsed,
        ))
    return results
