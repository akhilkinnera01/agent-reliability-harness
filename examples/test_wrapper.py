#!/usr/bin/env python3
"""
Test script for ARH Agent Wrapper

Demonstrates how to use the Agent Wrapper to query LLM endpoints.
Requires OPENAI_API_KEY environment variable for OpenAI tests.
"""

import os
import sys

# Add project root to path so we can import arh
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Import ARH components
from arh.core.agent_wrapper import OpenAIWrapper, OllamaWrapper, AgentWrapper
from arh.core.models import AgentResponse


def test_openai():
    """Test the OpenAI wrapper."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Skipping OpenAI test.")
        print("   Set the environment variable to test OpenAI integration.")
        return None
    
    print("🔄 Testing OpenAI wrapper...")
    wrapper = OpenAIWrapper(api_key=api_key)
    response = wrapper.query("What is 2 + 2?")
    
    print(f"✅ Response: {response.content}")
    print(f"⏱️  Latency: {response.latency_ms:.2f}ms")
    
    if response.error:
        print(f"❌ Error: {response.error}")
    
    return response


def test_ollama():
    """Test the Ollama wrapper (requires local Ollama instance)."""
    print("\n🔄 Testing Ollama wrapper...")
    wrapper = OllamaWrapper(model="llama2")
    response = wrapper.query("What is 2 + 2?")
    
    if response.error:
        print(f"⚠️  Ollama not available: {response.error}")
        print("   Start Ollama locally to test local model integration.")
        return None
    
    print(f"✅ Response: {response.content}")
    print(f"⏱️  Latency: {response.latency_ms:.2f}ms")
    
    return response


def test_custom_endpoint():
    """Demonstrate custom endpoint configuration."""
    print("\n📋 Custom Endpoint Example:")
    print("   wrapper = AgentWrapper(")
    print('       endpoint="https://your-api.com/v1/chat",')
    print('       auth_header={"Authorization": "Bearer YOUR_KEY"},')
    print('       model="your-model"')
    print("   )")
    print("   response = wrapper.query('Your prompt here')")


def main():
    """Run all tests."""
    print("=" * 60)
    print("ARH Agent Wrapper Test Suite")
    print("=" * 60)
    
    # Test imports
    print("\n✅ Imports successful!")
    print(f"   AgentResponse: {AgentResponse}")
    print(f"   OpenAIWrapper: {OpenAIWrapper}")
    print(f"   OllamaWrapper: {OllamaWrapper}")
    
    # Test OpenAI
    test_openai()
    
    # Test Ollama
    test_ollama()
    
    # Show custom endpoint example
    test_custom_endpoint()
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
