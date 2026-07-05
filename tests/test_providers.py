import pytest
import httpx
from unittest.mock import patch, MagicMock

from tokentriage.providers import _ollama_generate, _openrouter_generate

def test_retry_on_timeout():
    """Verify exponential backoff retries transient errors up to max_retries."""
    mock_post = MagicMock()
    # Fail twice with timeout, succeed on third attempt
    mock_post.side_effect = [
        httpx.TimeoutException("timeout 1"),
        httpx.TimeoutException("timeout 2"),
        MagicMock(
            status_code=200, 
            json=lambda: {"message": {"content": "success"}, "prompt_eval_count": 1, "eval_count": 1}
        )
    ]
    
    with patch("tokentriage.providers.httpx.post", mock_post):
        with patch("time.sleep") as mock_sleep: # Don't actually sleep in tests
            text, in_tok, out_tok = _ollama_generate("qwen2.5:3b", [{"role": "user", "content": "hello"}])
            
            assert text == "success"
            assert mock_post.call_count == 3
            assert mock_sleep.call_count == 2
            # Verify exponential backoff multiplier
            mock_sleep.assert_any_call(1.0)
            mock_sleep.assert_any_call(2.0)


def test_fail_fast_on_401():
    """Verify non-transient HTTP errors (like 401 Unauthorized) are not retried."""
    mock_post = MagicMock()
    
    # 401 is not in the retry list (429, 500, 502, 503, 504)
    request = httpx.Request("POST", "http://fake")
    response = httpx.Response(status_code=401, request=request)
    mock_post.side_effect = httpx.HTTPStatusError("unauthorized", request=request, response=response)
    
    with patch("tokentriage.providers.httpx.post", mock_post):
        with patch("time.sleep") as mock_sleep:
            with pytest.raises(httpx.HTTPStatusError):
                _ollama_generate("qwen2.5:3b", [{"role": "user", "content": "hello"}])
                
            assert mock_post.call_count == 1
            mock_sleep.assert_not_called()


def test_retry_on_429():
    """Verify transient HTTP errors (like 429 Rate Limit) are retried."""
    mock_post = MagicMock()
    
    request = httpx.Request("POST", "http://fake")
    response = httpx.Response(status_code=429, request=request)
    
    mock_post.side_effect = [
        httpx.HTTPStatusError("rate limited", request=request, response=response),
        MagicMock(
            status_code=200, 
            json=lambda: {"message": {"content": "success_after_429"}, "prompt_eval_count": 1, "eval_count": 1}
        )
    ]
    
    with patch("tokentriage.providers.httpx.post", mock_post):
        with patch("time.sleep") as mock_sleep:
            text, in_tok, out_tok = _ollama_generate("qwen2.5:3b", [{"role": "user", "content": "hello"}])
            
            assert text == "success_after_429"
            assert mock_post.call_count == 2
            assert mock_sleep.call_count == 1


def test_openrouter_uses_openai_compatible_base_url(monkeypatch):
    """OpenRouter dispatch should use the OpenAI SDK with OpenRouter base URL."""
    calls = {}

    class DummyUsage:
        prompt_tokens = 4
        completion_tokens = 5

    class DummyMessage:
        content = "rescued answer"

    class DummyChoice:
        message = DummyMessage()

    class DummyCompletions:
        def create(self, model, messages):
            calls["model"] = model
            calls["messages"] = messages
            return type("Resp", (), {"choices": [DummyChoice()], "usage": DummyUsage()})()

    class DummyClient:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.chat = type("Chat", (), {"completions": DummyCompletions()})()

    monkeypatch.setattr("openai.OpenAI", DummyClient)
    text, itok, otok = _openrouter_generate(
        "google/gemma-4-31b-it:free",
        "or-key",
        [{"role": "user", "content": "fix this"}],
    )

    assert text == "rescued answer"
    assert (itok, otok) == (4, 5)
    assert calls["model"] == "google/gemma-4-31b-it:free"
    assert str(calls["client"]["base_url"]).rstrip("/") == "https://openrouter.ai/api/v1"


def test_openrouter_zero_usage_falls_back_to_estimated_tokens(monkeypatch):
    """Free/provider-routed responses may report zero usage; receipts still need a baseline."""

    class ZeroUsage:
        prompt_tokens = 0
        completion_tokens = 0

    class DummyMessage:
        content = "This is a non-empty OpenRouter answer with enough text for token estimation."

    class DummyChoice:
        message = DummyMessage()

    class DummyCompletions:
        def create(self, model, messages):
            return type("Resp", (), {"choices": [DummyChoice()], "usage": ZeroUsage()})()

    class DummyClient:
        def __init__(self, **kwargs):
            self.chat = type("Chat", (), {"completions": DummyCompletions()})()

    monkeypatch.setattr("openai.OpenAI", DummyClient)
    text, itok, otok = _openrouter_generate(
        "openai/gpt-oss-20b:free",
        "or-key",
        [{"role": "user", "content": "Compare SOC 2, ISO 27001, and GDPR obligations."}],
    )

    assert text
    assert itok > 0
    assert otok > 0
