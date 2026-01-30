import pytest
from typing import Dict, Any

@pytest.fixture
def mock_openai_response() -> Dict[str, Any]:
    """Returns a fake OpenAI response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a fake OpenAI response."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        }
    }

@pytest.fixture
def mock_anthropic_response() -> Dict[str, Any]:
    """Returns a fake Anthropic response."""
    return {
        "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "This is a fake Anthropic response."
            }
        ],
        "model": "claude-3-haiku-20240307",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 12
        }
    }

@pytest.fixture
def mock_gemini_response() -> Dict[str, Any]:
    """Returns a fake Gemini response."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": "This is a fake Gemini response."
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE"
                    }
                ]
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 10,
            "totalTokenCount": 15
        }
    }
