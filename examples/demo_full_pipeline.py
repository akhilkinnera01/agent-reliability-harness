#!/usr/bin/env python3
"""
Full ARH Demonstration: Agent Testing + Document Auditing

This script demonstrates the complete ARH pipeline:
1. Agent Reliability Testing (4 dimensions)
2. Documentation Auditing (adversarial question generation)
3. Combined Trust Assessment

Automatically detects available API keys and uses the appropriate model.
Supports: Gemini, OpenAI, Anthropic, or falls back to mock agent.
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

# Import ARH components
from arh.core.agent_wrapper import AgentWrapper
from arh.core.harness import ReliabilityHarness
from arh.core.models import AgentResponse
from arh.auditor.auditor import AdversarialAuditor
from arh.metrics import MetricsExporter


# Sample document with intentional flaws
SAMPLE_DOC = """
## Chemical Handling Procedures

When handling corrosive substances, ensure proper ventilation.
Transfer chemicals using appropriate containers.
In case of spills, follow cleanup procedures immediately.
Always wear protective equipment when in the lab.

## Equipment Operation

Before using any equipment, complete the required training.
Follow manufacturer guidelines for all operations.
Report any malfunctions to your supervisor.
"""


def detect_and_create_agent():
    """
    Auto-detect available API keys and create the appropriate agent.
    
    Priority order:
    1. GEMINI_API_KEY -> Gemini 2.5 Flash
    2. OPENAI_API_KEY -> GPT-4o-mini
    3. ANTHROPIC_API_KEY -> Claude 3.5 Sonnet
    4. Fallback -> Mock agent
    """
    from arh.core import UniversalWrapper
    
    # Check for Gemini
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print("✅ Found GEMINI_API_KEY - using Gemini 2.5 Flash")
        return UniversalWrapper(
            model="gemini/gemini-2.5-flash",
            api_key=gemini_key
        ), "gemini-2.5-flash"
    
    # Check for OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("✅ Found OPENAI_API_KEY - using GPT-4o-mini")
        return UniversalWrapper(
            model="gpt-4o-mini",
            api_key=openai_key
        ), "gpt-4o-mini"
    
    # Check for Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        print("✅ Found ANTHROPIC_API_KEY - using Claude 3.5 Sonnet")
        return UniversalWrapper(
            model="anthropic/claude-3-5-sonnet-20241022",
            api_key=anthropic_key
        ), "claude-3.5-sonnet"
    
    # Check for Groq (free tier available)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print("✅ Found GROQ_API_KEY - using Llama 3.1 70B")
        return UniversalWrapper(
            model="groq/llama-3.1-70b-versatile",
            api_key=groq_key
        ), "llama-3.1-70b"
    
    # Fallback to mock
    print("⚠️  No API key found - using mock agent")
    print("   Set one of: GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY")
    return MockAgent(), "mock"


class MockAgent(AgentWrapper):
    """Mock agent for demo without API keys."""
    
    def __init__(self):
        super().__init__(endpoint="mock://", model="mock-demo")
        self.call_count = 0
    
    def query(self, prompt: str, **kwargs) -> AgentResponse:
        import time
        import random
        
        self.call_count += 1
        start = time.time()
        
        # Simulate latency
        time.sleep(random.uniform(0.05, 0.15))
        
        # Generate contextual responses
        if "2 + 2" in prompt or "2+2" in prompt:
            content = "The answer is 4."
        elif "capital" in prompt.lower() and "france" in prompt.lower():
            content = "The capital of France is Paris."
        elif "confidence" in prompt.lower():
            content = "My confidence is 85%. I am fairly certain."
        else:
            content = f"Response to query {self.call_count}"
        
        latency = (time.time() - start) * 1000
        return AgentResponse(content=content, latency_ms=latency, model=self.model)


def run_agent_reliability_test(agent, model_name):
    """Part 1: Test agent reliability across 4 dimensions."""
    print("=" * 60)
    print("PART 1: Agent Reliability Testing")
    print("=" * 60)
    
    harness = ReliabilityHarness(agent)
    
    test_prompts = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "Explain quantum computing in simple terms."
    ]
    
    print(f"Testing with {len(test_prompts)} prompts using {model_name}...")
    harness.run_all(test_prompts)
    
    report = harness.generate_report()
    
    print(f"\n📊 Agent: {model_name}")
    print(f"   Overall Score: {report['overall_score']:.1%}")
    print(f"   Verdict: {report['verdict']}")
    print("\n   Dimensions:")
    
    for name, data in report.get("dimensions", {}).items():
        status = "✅" if data["status"] == "pass" else "❌"
        print(f"   {status} {name.capitalize()}: {data['score']:.1%}")
    
    return report


def run_documentation_audit(agent, model_name):
    """Part 2: Audit documentation for flaws."""
    print("\n" + "=" * 60)
    print("PART 2: Documentation Audit")
    print("=" * 60)
    
    auditor = AdversarialAuditor(proposer_model=agent)
    
    print(f"Auditing document ({len(SAMPLE_DOC)} characters) using {model_name}...")
    
    # Use simple mode if mock, otherwise full LLM mode
    if model_name == "mock":
        report = auditor.audit_simple(SAMPLE_DOC, document_name="chemical_procedures.md")
    else:
        report = auditor.audit(SAMPLE_DOC, document_name="chemical_procedures.md")
    
    print(f"\n📋 Document: {report.document}")
    print(f"   Score: {report.overall_score:.1%}")
    print(f"   Findings: {len(report.findings)}")
    
    if report.findings:
        print("\n   Top Findings:")
        for i, finding in enumerate(report.findings[:3], 1):
            print(f"   {i}. [{finding.severity.value.upper()}] {finding.flaw_type.value}")
            print(f"      Question: {finding.question[:50]}...")
            print(f"      Fix: {finding.recommendation[:60]}...")
    
    return report


def run_combined_trust_assessment(agent_report, audit_report):
    """Part 3: Calculate combined trust score."""
    print("\n" + "=" * 60)
    print("PART 3: Combined Trust Assessment")
    print("=" * 60)
    
    agent_score = agent_report.get("overall_score", 0)
    doc_score = audit_report.overall_score
    
    # Weighted combination: 60% agent, 40% knowledge
    trust_score = 0.6 * agent_score + 0.4 * doc_score
    
    if trust_score >= 0.85:
        verdict = "PASS"
        color = "✅"
    elif trust_score >= 0.70:
        verdict = "CONDITIONAL_PASS"
        color = "⚠️"
    else:
        verdict = "BLOCK"
        color = "❌"
    
    print(f"\n   Agent Reliability: {agent_score:.1%}")
    print(f"   Documentation Quality: {doc_score:.1%}")
    print(f"   ─────────────────────────")
    print(f"   Combined Trust Score: {trust_score:.1%}")
    print(f"   {color} Deployment Verdict: {verdict}")
    
    return {
        "agent_score": agent_score,
        "doc_score": doc_score,
        "trust_score": trust_score,
        "verdict": verdict
    }


def export_metrics(agent_report, audit_report, trust_result, model_name):
    """Part 4: Export metrics for monitoring."""
    print("\n" + "=" * 60)
    print("PART 4: Metrics Export")
    print("=" * 60)
    
    exporter = MetricsExporter(system_name="demo-system")
    
    exporter.export_agent_results(model_name, agent_report)
    exporter.export_audit_results("demo-doc", audit_report)
    exporter.export_trust_score(
        trust_result["agent_score"],
        trust_result["doc_score"]
    )
    
    snapshot = exporter.get_snapshot_dict()
    print("\n   Metrics Snapshot:")
    print(f"   - Trust Score: {snapshot['trust_score']:.1%}")
    print(f"   - Deployment Ready: {snapshot['deployment_ready']}")
    
    print("\n   Sample Prometheus Output:")
    metrics_output = exporter.get_metrics().decode()[:500]
    for line in metrics_output.split('\n')[:5]:
        if line and not line.startswith('#'):
            print(f"   {line}")


def main():
    """Run the full ARH demonstration."""
    print("\n" + "🚀" + " " + "=" * 56 + " " + "🚀")
    print("   AGENT RELIABILITY HARNESS (ARH) - FULL DEMO")
    print("🚀" + " " + "=" * 56 + " " + "🚀")
    
    # Auto-detect API key and create agent
    print("\n🔍 Detecting available API keys...")
    agent, model_name = detect_and_create_agent()
    
    # Run all parts
    agent_report = run_agent_reliability_test(agent, model_name)
    audit_report = run_documentation_audit(agent, model_name)
    trust_result = run_combined_trust_assessment(agent_report, audit_report)
    export_metrics(agent_report, audit_report, trust_result, model_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"\n🤖 Model Used: {model_name}")
    print(f"🎯 Final Trust Score: {trust_result['trust_score']:.1%}")
    print(f"📋 Verdict: {trust_result['verdict']}")
    print("\nFor more information, see:")
    print("  - docs/GETTING-STARTED.md")
    print("  - docs/DRZERO_CONNECTION.md")
    print("  - README.md")


if __name__ == "__main__":
    main()
