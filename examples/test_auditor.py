#!/usr/bin/env python3
"""
Test script for ARH Adversarial Auditor

Demonstrates how to audit documents for flaws using the adversarial auditor.
Can test with OpenAI or in simple mode (no API required).
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

# Import ARH components
from arh.core.agent_wrapper import OpenAIWrapper, AgentWrapper
from arh.core.models import AgentResponse
from arh.auditor import AdversarialAuditor, HopComplexity
import json


# Sample document with intentional flaws for testing
SAMPLE_DOCUMENT = """
# Lab Safety Manual

## 1. Introduction
This document outlines the safety procedures for our laboratory.
All personnel must follow these guidelines.

## 2. Chemical Handling
When working with chemicals, ensure proper ventilation.
Handle all substances carefully. Wear appropriate protection.
Store chemicals in designated areas.

## 3. Equipment Operation
Before using equipment, complete the required training.
Follow manufacturer guidelines.
Report any malfunctions immediately to the supervisor.

## 4. Emergency Procedures
In case of emergency, contact the safety officer.
Know the location of exits and safety equipment.
"""


class MockAgentForAudit(AgentWrapper):
    """Mock agent that generates structured responses for auditing demo."""
    
    def __init__(self):
        super().__init__(endpoint="mock://localhost", model="mock-auditor")
    
    def query(self, prompt: str, **kwargs) -> AgentResponse:
        import time
        start = time.time()
        
        # Simulate response based on prompt type
        if "adversarial documentation auditor" in prompt.lower():
            # Proposer prompt - generate questions
            content = """
Q1: What specific CFM rate is required for ventilation?
TARGET: ventilation rate specification
FLAW_IF_MISSING: AMBIGUOUS

Q2: What PPE is required for acid handling?
TARGET: PPE requirements for acids
FLAW_IF_MISSING: MISSING_PREREQ

Q3: What is the emergency phone number?
TARGET: emergency contact number
FLAW_IF_MISSING: MISSING_PREREQ

Q4: What happens if the supervisor is unavailable?
TARGET: backup contact procedure
FLAW_IF_MISSING: SAFETY_GAP

Q5: How long is equipment training?
TARGET: training duration
FLAW_IF_MISSING: IMPLICIT_ASSUMPTION
"""
        elif "STRICT documentation validator" in prompt:
            # Solver prompt - answer from document
            if "CFM" in prompt or "ventilation rate" in prompt:
                content = """
STATUS: NOT_FOUND
CONFIDENCE: 10
ANSWER: Cannot determine from document
CITATION: N/A
MISSING: Specific ventilation rate in CFM or air changes per hour
"""
            elif "PPE" in prompt:
                content = """
STATUS: AMBIGUOUS
CONFIDENCE: 30
ANSWER: Document says "appropriate protection" but doesn't specify
CITATION: "Wear appropriate protection"
MISSING: Specific PPE requirements for different chemical types
"""
            elif "phone" in prompt or "number" in prompt:
                content = """
STATUS: NOT_FOUND
CONFIDENCE: 5
ANSWER: Cannot determine from document
CITATION: N/A
MISSING: Contact phone number for safety officer
"""
            else:
                content = """
STATUS: PARTIAL
CONFIDENCE: 40
ANSWER: Some related information exists but is incomplete
CITATION: N/A
MISSING: More specific details
"""
        else:
            content = "Mock response for testing"
        
        latency = (time.time() - start) * 1000
        return AgentResponse(content=content, latency_ms=latency, model=self.model)


def run_simple_audit():
    """Run audit using simple keyword matching (no LLM required)."""
    print("\n🔍 Simple Audit Demo (No LLM Required)")
    print("-" * 40)
    
    agent = MockAgentForAudit()
    auditor = AdversarialAuditor(proposer_model=agent)
    
    # Use the simple audit method
    report = auditor.audit_simple(SAMPLE_DOCUMENT, document_name="lab_safety_manual.md")
    
    auditor.print_report(report)
    
    return report


def run_mock_audit():
    """Run audit using mock agent that simulates LLM responses."""
    print("\n🎭 Mock Audit Demo")
    print("-" * 40)
    
    agent = MockAgentForAudit()
    auditor = AdversarialAuditor(
        proposer_model=agent,
        hop_complexity=[HopComplexity.ONE, HopComplexity.TWO]
    )
    
    report = auditor.audit(SAMPLE_DOCUMENT, document_name="lab_safety_manual.md")
    
    auditor.print_report(report)
    
    # Show JSON report
    print("\n📋 Full Report (JSON):")
    report_dict = auditor.generate_report_dict(report)
    print(json.dumps(report_dict, indent=2))
    
    return report


def run_openai_audit():
    """Run real audit with OpenAI (requires API key)."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("\n⚠️  OPENAI_API_KEY not set. Skipping OpenAI audit.")
        return None
    
    print("\n🤖 OpenAI Audit Demo")
    print("-" * 40)
    
    agent = OpenAIWrapper(api_key=api_key)
    auditor = AdversarialAuditor(
        proposer_model=agent,
        hop_complexity=[HopComplexity.ONE]  # Just 1-hop to save API costs
    )
    
    # Audit just a small section
    small_doc = SAMPLE_DOCUMENT[:500]
    report = auditor.audit(small_doc, document_name="lab_safety_excerpt.md")
    
    auditor.print_report(report)
    
    return report


def main():
    """Run all demos."""
    print("=" * 60)
    print("ARH Adversarial Auditor Test Suite")
    print("=" * 60)
    
    # Test imports
    print("\n✅ All imports successful!")
    
    # Show sample document
    print("\n📄 Sample Document (with intentional flaws):")
    print("-" * 40)
    print(SAMPLE_DOCUMENT[:300] + "...")
    
    # Run demos
    run_simple_audit()
    run_mock_audit()
    run_openai_audit()
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
