import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from tokentriage.proxy.app import app

client = TestClient(app)

def test_judge_mode_gating():
    """Verify that demo replay endpoints return 404 when judge-mode is disabled."""
    
    # By default in tests, TOKENTRIAGE_JUDGE_MODE is not set, so _judge_mode is False
    with patch("tokentriage.proxy.app._judge_mode", False):
        response = client.post("/demo/replay/reset")
        assert response.status_code == 404
        assert response.json() == {"detail": "demo replay unavailable"}
        
        response = client.get("/demo/replay/list")
        assert response.status_code == 404

        response = client.get("/demo/replay/questions")
        assert response.status_code == 404
        
        response = client.get("/demo/replay/item/0")
        assert response.status_code == 404
        
        response = client.get("/demo/replay/next")
        assert response.status_code == 404
        
        response = client.get("/demo/replay/prev")
        assert response.status_code == 404


def test_judge_mode_enabled():
    """Verify that demo replay endpoints are accessible when judge-mode is enabled."""
    
    with patch("tokentriage.proxy.app._judge_mode", True):
        response = client.get("/demo/replay/list")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 6
        assert data["items"][0]["task_preview"].startswith("What is the capital")

        response = client.get("/demo/replay/questions")
        assert response.status_code == 200
        questions = response.json()
        assert questions["count"] == 3
        assert questions["items"][0]["idx"] == 0
        assert "Australia" in questions["items"][0]["question"]
        assert questions["items"][1]["idx"] == 2
        assert "vendor" in questions["items"][1]["question"].lower()
        assert questions["items"][2]["idx"] == 3
        assert "GDPR" in questions["items"][2]["question"]

        response = client.get("/demo/replay/next")
        assert response.status_code == 200
        item = response.json()
        assert item["index"] == 0
        assert item["conversation"][0]["role"] == "user"
        assert item["conversation"][1]["role"] == "assistant"
        assert "Canberra" in item["conversation"][1]["content"]

        response = client.get("/demo/replay/prev")
        assert response.status_code == 200
        item = response.json()
        assert item["index"] == data["count"] - 1
        assert item["conversation"][1]["content"].startswith("Blocked:")
