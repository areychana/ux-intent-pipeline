from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@patch("api.main.run_pipeline")
def test_analyze_single(mock_pipeline):
    mock_pipeline.return_value = {
        "intent": "navigation_issue",
        "confidence": 0.87,
        "topics": ["menu", "sidebar"],
        "summary": "User struggled to find the settings menu.",
    }
    response = client.post("/analyze", json={"text": "I can't find the settings", "source": "survey"})
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "navigation_issue"
    assert data["confidence"] == 0.87
    assert "processing_time_ms" in data


@patch("api.main.detect_intent")
def test_intent_only(mock_detect):
    mock_detect.return_value = ("usability_complaint", 0.91)
    response = client.post("/intent", json={"text": "This button is confusing"})
    assert response.status_code == 200
    assert response.json()["intent"] == "usability_complaint"


@patch("api.main.extract_topics")
def test_topics_batch(mock_topics):
    mock_topics.return_value = [
        {"keywords": ["login", "password"], "texts": ["Can't log in"]},
        {"keywords": ["slow", "loading"], "texts": ["Takes forever to load"]},
    ]
    response = client.post("/topics", json={
        "items": [
            {"text": "Can't log in"},
            {"text": "Takes forever to load"},
        ]
    })
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_topics_requires_minimum_two():
    response = client.post("/topics", json={"items": [{"text": "only one item"}]})
    assert response.status_code == 422
