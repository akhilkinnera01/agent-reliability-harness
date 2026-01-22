#!/usr/bin/env python3
"""
Test script for ARH Reliability Harness

Demonstrates how to run reliability tests on an LLM agent.
Can test with OpenAI, Ollama, or a mock agent for demonstration.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

# Import ARH components
from arh.core.agent_wrapper import OpenAIWrapper, OllamaWrapper, AgentWrapper
from arh.core.harness import ReliabilityHarness
from arh.core.models import AgentResponse
import json


class MockAgent(AgentWrapper):
    """
    A mock agent for testing without API access.
    Returns predictable responses for demonstration.
    """
    
    def __init__(self):
        super().__init__(endpoint="mock://localhost", model="mock-agent")
        self.call_count = 0
    
    def query(self, prompt: str, **kwargs) -> AgentResponse:
        """Return a mock response."""
        import time
        import random
        
        self.call_count += 1
        start_time = time.time()
        
        # Simulate some latency
        latency_ms = random.uniform(50, 200)
        time.sleep(latency_ms / 1000)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Generate a simple response based on the prompt
        if "2 + 2" in prompt.lower() or "2+2" in prompt.lower():
            content = "The answer is 4."
        elif "capital" in prompt.lower() and "france" in prompt.lower():
            content = "The capital of France is Paris."
        elif "confidence" in prompt.lower():
            content = "My confidence is 85%. I am fairly certain about this."
        else:
            content = f"This is a mock response to: {prompt[:50]}..."
        
        result = AgentResponse(
            content=content,
            latency_ms=latency_ms,
            model=self.model
        )
        
        self.response_log.append(result)
        return result


def run_mock_demo():
    """Run a demonstration with the mock agent."""
    print("\n🎭 Running Mock Agent Demo")
    print("-" * 40)
    
    # Create mock agent and harness
    agent = MockAgent()
    harness = ReliabilityHarness(agent)
    
    # Define test prompts
    prompts = [
        "What is 2 + 2?",
        "What is the capital of France?",
        "Explain quantum computing briefly."
    ]
    
    # Run all tests
    print("Running reliability tests...")
    results = harness.run_all(prompts)
    
    # Print summary
    harness.print_summary()
    
    # Generate and display report
    report = harness.generate_report()
    print("\n📋 Full Report (JSON):")
    print(json.dumps(report, indent=2, default=str))
    
    return report


def run_openai_demo():
    """Run tests with OpenAI (requires API key)."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Skipping OpenAI demo.")
        return None
    
    print("\n🤖 Running OpenAI Demo")
    print("-" * 40)
    
    agent = OpenAIWrapper(api_key=api_key)
    harness = ReliabilityHarness(agent)
    
    prompts = [
        "What is 2 + 2?",
        "What is the capital of France?",
    ]
    
    # Run just predictability to save API costs
    print("Running predictability test (to save API costs)...")
    result = harness.run_test("predictability", prompts, samples=5)
    
    print(f"✅ Predictability Score: {result.score:.1%}")
    print(f"   P50: {result.details.get('p50_ms', 0):.0f}ms")
    print(f"   P99: {result.details.get('p99_ms', 0):.0f}ms")
    
    return result


def run_individual_test_demo():
    """Demonstrate running individual tests."""
    print("\n🔬 Individual Test Demo")
    print("-" * 40)
    
    agent = MockAgent()
    harness = ReliabilityHarness(agent)
    
    prompts = ["What is 2 + 2?", "What is the capital of France?"]
    
    # Run just robustness with custom settings
    result = harness.run_test(
        "robustness",
        prompts,
        perturbations=["typo", "case_shift"],
        threshold=0.7
    )
    
    print(f"Robustness Test Result:")
    print(f"  Score: {result.score:.1%}")
    print(f"  Status: {result.status.value}")
    print(f"  Details: {result.details}")


def main():
    """Run all demos."""
    print("=" * 60)
    print("ARH Reliability Harness Test Suite")
    print("=" * 60)
    
    # Test imports
    print("\n✅ All imports successful!")
    
    # Run demos
    run_mock_demo()
    run_individual_test_demo()
    run_openai_demo()
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
