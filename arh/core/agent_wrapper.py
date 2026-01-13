"""
ARH Agent Wrapper

Universal wrapper for testing any LLM agent endpoint.
Supports OpenAI, Anthropic, local models (Ollama), and custom APIs.
Provides consistent interface for querying and measuring response metrics.
"""

import httpx
import time
from typing import Optional, Dict, List, Any
from .models import AgentResponse


class AgentWrapper:
    """
    Universal wrapper for testing any LLM agent endpoint.
    Supports OpenAI, Anthropic, local models, and custom APIs.
    """
    
    def __init__(
        self,
        endpoint: str,
        auth_header: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        model: str = "unknown"
    ):
        """
        Initialize the agent wrapper.
        
        Args:
            endpoint: The API endpoint URL
            auth_header: Optional authentication headers
            timeout: Request timeout in seconds
            model: Model identifier for logging
        """
        self.endpoint = endpoint
        self.auth_header = auth_header or {}
        self.timeout = timeout
        self.model = model
        self.response_log: List[AgentResponse] = []
    
    def query(self, prompt: str, **kwargs) -> AgentResponse:
        """
        Send a query to the agent and return normalized response.
        
        Args:
            prompt: The prompt to send to the agent
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            AgentResponse with content, latency, and metadata
        """
        start_time = time.time()
        
        try:
            # Build request based on endpoint type
            payload = self._build_payload(prompt, **kwargs)
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.endpoint,
                    json=payload,
                    headers=self.auth_header
                )
                response.raise_for_status()
            
            latency_ms = (time.time() - start_time) * 1000
            content = self._extract_content(response.json())
            
            result = AgentResponse(
                content=content,
                latency_ms=latency_ms,
                model=self.model,
                metadata=response.json()
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            result = AgentResponse(
                content="",
                latency_ms=latency_ms,
                model=self.model,
                error=str(e)
            )
        
        self.response_log.append(result)
        return result
    
    def batch_query(self, prompts: List[str], **kwargs) -> List[AgentResponse]:
        """
        Query multiple prompts sequentially.
        
        Args:
            prompts: List of prompts to send
            **kwargs: Additional parameters passed to each query
            
        Returns:
            List of AgentResponse objects
        """
        return [self.query(p, **kwargs) for p in prompts]
    
    def _build_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Build API payload. Override for custom formats.
        
        Default format is OpenAI-compatible.
        """
        return {
            "model": kwargs.get("model", self.model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
    
    def _extract_content(self, response: Dict[str, Any]) -> str:
        """
        Extract content from response. Override for custom formats.
        
        Handles OpenAI and Anthropic formats by default.
        """
        # OpenAI format
        if "choices" in response:
            return response["choices"][0]["message"]["content"]
        # Anthropic format
        if "content" in response:
            return response["content"][0]["text"]
        # Generic fallback
        return str(response)
    
    def clear_log(self):
        """Clear the response log."""
        self.response_log = []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the response log.
        
        Returns:
            Dictionary with count, average latency, and error rate
        """
        if not self.response_log:
            return {"count": 0, "avg_latency_ms": 0, "error_rate": 0}
        
        latencies = [r.latency_ms for r in self.response_log]
        errors = sum(1 for r in self.response_log if r.error)
        
        return {
            "count": len(self.response_log),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "error_rate": errors / len(self.response_log)
        }


class OpenAIWrapper(AgentWrapper):
    """Pre-configured wrapper for OpenAI API."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI wrapper.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini)
        """
        super().__init__(
            endpoint="https://api.openai.com/v1/chat/completions",
            auth_header={"Authorization": f"Bearer {api_key}"},
            model=model
        )


class AnthropicWrapper(AgentWrapper):
    """Pre-configured wrapper for Anthropic API."""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        """
        Initialize Anthropic wrapper.
        
        Args:
            api_key: Anthropic API key
            model: Model to use (default: claude-3-haiku-20240307)
        """
        super().__init__(
            endpoint="https://api.anthropic.com/v1/messages",
            auth_header={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            model=model
        )
    
    def _build_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Build Anthropic-specific payload format."""
        return {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "messages": [{"role": "user", "content": prompt}]
        }


class OllamaWrapper(AgentWrapper):
    """Pre-configured wrapper for local Ollama instance."""
    
    def __init__(self, model: str = "llama2", host: str = "localhost", port: int = 11434):
        """
        Initialize Ollama wrapper.
        
        Args:
            model: Model to use (default: llama2)
            host: Ollama host (default: localhost)
            port: Ollama port (default: 11434)
        """
        super().__init__(
            endpoint=f"http://{host}:{port}/api/generate",
            model=model
        )
    
    def _build_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Build Ollama-specific payload format."""
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
    
    def _extract_content(self, response: Dict[str, Any]) -> str:
        """Extract content from Ollama response."""
        return response.get("response", "")


class GeminiWrapper(AgentWrapper):
    """Pre-configured wrapper for Google Gemini API."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        """
        Initialize Gemini wrapper.
        
        Args:
            api_key: Google AI API key
            model: Model to use (default: gemini-2.0-flash)
                   Options: gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash
        """
        self.api_key = api_key
        super().__init__(
            endpoint=f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            auth_header={},  # API key is passed as query param
            model=model
        )
    
    def query(self, prompt: str, **kwargs) -> AgentResponse:
        """Send a query to Gemini API."""
        start_time = time.time()
        
        try:
            payload = self._build_payload(prompt, **kwargs)
            
            # Gemini uses API key as query parameter
            url = f"{self.endpoint}?key={self.api_key}"
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
            
            latency_ms = (time.time() - start_time) * 1000
            content = self._extract_content(response.json())
            
            result = AgentResponse(
                content=content,
                latency_ms=latency_ms,
                model=self.model,
                metadata=response.json()
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            result = AgentResponse(
                content="",
                latency_ms=latency_ms,
                model=self.model,
                error=str(e)
            )
        
        self.response_log.append(result)
        return result
    
    def _build_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Build Gemini-specific payload format."""
        return {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 1000),
            }
        }
    
    def _extract_content(self, response: Dict[str, Any]) -> str:
        """Extract content from Gemini response."""
        try:
            return response["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return str(response)


class UniversalWrapper(AgentWrapper):
    """
    Universal wrapper using LiteLLM - supports 100+ models!
    
    Model format examples:
        - OpenAI: "gpt-4o", "gpt-4o-mini"
        - Gemini: "gemini/gemini-2.5-flash", "gemini/gemini-2.0-flash"
        - Anthropic: "anthropic/claude-3-5-sonnet"
        - Ollama: "ollama/llama3"
        - Groq: "groq/llama-3.1-70b-versatile"
        - Mistral: "mistral/mistral-large-latest"
    
    See full list: https://docs.litellm.ai/docs/providers
    """
    
    def __init__(
        self,
        model: str = "gemini/gemini-2.5-flash",
        api_key: str = None,
        **kwargs
    ):
        """
        Initialize universal wrapper.
        
        Args:
            model: Model identifier in LiteLLM format (e.g., "gemini/gemini-2.5-flash")
            api_key: API key (or use environment variables)
            **kwargs: Additional parameters passed to litellm.completion
        """
        super().__init__(endpoint="litellm", model=model)
        self.api_key = api_key
        self.extra_params = kwargs
        
        # Check if litellm is available
        try:
            import litellm
            self.litellm = litellm
        except ImportError:
            raise ImportError(
                "LiteLLM is required for UniversalWrapper. "
                "Install it with: pip install litellm"
            )
    
    def query(self, prompt: str, **kwargs) -> AgentResponse:
        """Send a query using LiteLLM."""
        start_time = time.time()
        
        try:
            # Merge parameters
            params = {**self.extra_params, **kwargs}
            
            # Build messages
            messages = [{"role": "user", "content": prompt}]
            
            # Call LiteLLM
            response = self.litellm.completion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_tokens", 1000),
            )
            
            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content
            
            result = AgentResponse(
                content=content,
                latency_ms=latency_ms,
                model=self.model,
                metadata={
                    "usage": dict(response.usage) if response.usage else {},
                    "model": response.model,
                }
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            result = AgentResponse(
                content="",
                latency_ms=latency_ms,
                model=self.model,
                error=str(e)
            )
        
        self.response_log.append(result)
        return result
