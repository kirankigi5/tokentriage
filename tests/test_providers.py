import pytest
import httpx
from unittest.mock import patch, MagicMock

from tokentriage.providers import _ollama_generate

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
