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
        
        response = client.get("/demo/replay/item/0")
        assert response.status_code == 404
        
        response = client.get("/demo/replay/next")
        assert response.status_code == 404
        
        response = client.get("/demo/replay/prev")
        assert response.status_code == 404


def test_judge_mode_enabled():
    """Verify that demo replay endpoints are accessible when judge-mode is enabled."""
    
    with patch("tokentriage.proxy.app._judge_mode", True):
        # We mock out the database query so it doesn't crash on an empty DB
        with patch("tokentriage.proxy.app._decision_rows", return_value=[]):
            response = client.get("/demo/replay/list")
            assert response.status_code == 200
            assert response.json() == {"count": 0, "items": []}
            
            response = client.get("/demo/replay/next")
            assert response.status_code == 404
            assert response.json() == {"error": "no replay items seeded"}
