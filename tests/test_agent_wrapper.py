import pytest
import httpx
from typing import Dict, Any, List
from arh.core.agent_wrapper import AgentWrapper, OpenAIWrapper, AnthropicWrapper, GeminiWrapper, AgentResponse

# Test 1: AgentWrapper.query() success
def test_agent_wrapper_query_success(mocker, mock_openai_response):
    """Test successful query."""
    # Mock httpx.Client
    mock_response = mocker.Mock()
    mock_response.json.return_value = mock_openai_response
    mock_response.raise_for_status.return_value = None

    mock_client_instance = mocker.Mock()
    mock_client_instance.post.return_value = mock_response

    mock_client_cls = mocker.patch("arh.core.agent_wrapper.httpx.Client")
    mock_client_cls.return_value.__enter__.return_value = mock_client_instance

    wrapper = AgentWrapper(endpoint="http://fake-endpoint", model="fake-model")
    response = wrapper.query("test prompt")

    assert isinstance(response, AgentResponse)
    assert response.content == "This is a fake OpenAI response."
    assert response.latency_ms >= 0
    assert response.error is None
    assert response.model == "fake-model"
    assert response.metadata == mock_openai_response

    mock_client_instance.post.assert_called_once()

# Test 2: AgentWrapper.query() error handling
def test_agent_wrapper_query_error(mocker):
    """Test query error handling (timeout)."""
    # Mock httpx.Client to raise TimeoutException
    mock_client_instance = mocker.Mock()
    mock_client_instance.post.side_effect = httpx.TimeoutException("Request timed out")

    mock_client_cls = mocker.patch("arh.core.agent_wrapper.httpx.Client")
    mock_client_cls.return_value.__enter__.return_value = mock_client_instance

    wrapper = AgentWrapper(endpoint="http://fake-endpoint", model="fake-model")
    response = wrapper.query("test prompt")

    assert isinstance(response, AgentResponse)
    assert response.content == ""
    assert response.latency_ms >= 0
    assert response.error is not None
    assert "Request timed out" in response.error

# Test 3: AgentWrapper.batch_query()
def test_agent_wrapper_batch_query(mocker):
    """Test batch query calls query multiple times."""
    wrapper = AgentWrapper(endpoint="http://fake-endpoint")

    # Mock query method
    mock_query = mocker.patch.object(wrapper, 'query')
    # We need to return an object that matches the return type hint,
    # but for mocking purposes, any object is fine as long as we verify the calls.
    # However, let's return a proper AgentResponse to be safe.
    mock_query.return_value = AgentResponse(content="fake", latency_ms=10)

    prompts = ["p1", "p2", "p3"]
    responses = wrapper.batch_query(prompts, temperature=0.5)

    assert len(responses) == 3
    assert mock_query.call_count == 3

    # Verify call args
    mock_query.assert_any_call("p1", temperature=0.5)
    mock_query.assert_any_call("p2", temperature=0.5)
    mock_query.assert_any_call("p3", temperature=0.5)

# Test 4: AgentWrapper.get_stats()
def test_agent_wrapper_get_stats():
    """Test statistics calculation."""
    wrapper = AgentWrapper(endpoint="http://fake-endpoint")

    # Add fake responses to log
    wrapper.response_log = [
        AgentResponse(content="ok", latency_ms=100.0, error=None),
        AgentResponse(content="ok", latency_ms=200.0, error=None),
        AgentResponse(content="", latency_ms=300.0, error="some error"),
    ]

    stats = wrapper.get_stats()

    assert stats["count"] == 3
    assert stats["avg_latency_ms"] == 200.0  # (100+200+300)/3
    assert abs(stats["error_rate"] - 1/3) < 0.0001

    # Test empty log
    wrapper.clear_log()
    stats = wrapper.get_stats()
    assert stats["count"] == 0
    assert stats["avg_latency_ms"] == 0
    assert stats["error_rate"] == 0

# Test 5: OpenAIWrapper._build_payload()
def test_openai_wrapper_build_payload():
    """Test OpenAI payload format."""
    wrapper = OpenAIWrapper(api_key="fake-key")
    payload = wrapper._build_payload("test prompt", temperature=0.5, max_tokens=200)

    assert payload["model"] == "gpt-4o-mini"
    assert payload["messages"] == [{"role": "user", "content": "test prompt"}]
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 200

# Test 6: AnthropicWrapper._build_payload()
def test_anthropic_wrapper_build_payload():
    """Test Anthropic payload format."""
    wrapper = AnthropicWrapper(api_key="fake-key")
    payload = wrapper._build_payload("test prompt", max_tokens=500)

    assert payload["model"] == "claude-3-haiku-20240307"
    assert payload["messages"] == [{"role": "user", "content": "test prompt"}]
    assert payload["max_tokens"] == 500

# Test 7: GeminiWrapper._extract_content()
def test_gemini_wrapper_extract_content(mock_gemini_response):
    """Test Gemini content extraction."""
    wrapper = GeminiWrapper(api_key="fake-key")
    content = wrapper._extract_content(mock_gemini_response)

    assert content == "This is a fake Gemini response."

    # Test error handling in extraction
    assert wrapper._extract_content({}) == "{}"
