# Agent Reliability Harness - Getting Started Guide

This guide walks you through running the ARH reliability tests and adversarial auditor through Steps 1-3.

---

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/akhilkinnera/Documents/Projects/Agent-Reliability-Harness
source .venv/bin/activate  # If using virtual environment
pip3 install -r requirements.txt
```

### 2. Run Step 1: Basic Wrapper Test

```bash
python3 examples/test_wrapper.py
```

**What you'll see:**
```
============================================================
ARH Agent Wrapper Test Suite
============================================================

✅ Imports successful!
   AgentResponse: <class 'arh.core.models.AgentResponse'>
   OpenAIWrapper: <class 'arh.core.agent_wrapper.OpenAIWrapper'>
   OllamaWrapper: <class 'arh.core.agent_wrapper.OllamaWrapper'>
⚠️  OPENAI_API_KEY not set. Skipping OpenAI test.
⚠️  Ollama not available: [Errno 61] Connection refused

============================================================
Test suite completed!
============================================================
```

This confirms the core wrapper is working.

### 3. Run Step 2: Reliability Harness

```bash
python3 examples/test_harness.py
```

**What you'll see:**
```
============================================================
ARH Reliability Harness Test Suite
============================================================

✅ All imports successful!

🎭 Running Mock Agent Demo
----------------------------------------
RELIABILITY ASSESSMENT SUMMARY
Agent: mock-agent
Overall Score: 96.7%
Verdict: PASS
------------------------------------------------------------
✅ Robustness: 86.7%
   └─ truncate: 'What is 2 + 2?...' produced different output
✅ Consistency: 100.0%
✅ Groundedness: 100.0%
✅ Predictability: 100.0%
============================================================

📋 Full Report (JSON):
{ ... detailed report ... }
```

### 4. Run Step 3: Adversarial Auditor

```bash
python3 examples/test_auditor.py
```

**What you'll see:**
```
============================================================
ARH Adversarial Auditor Test Suite
============================================================

✅ All imports successful!

📄 Sample Document (with intentional flaws):
----------------------------------------
# Lab Safety Manual
## 1. Introduction
This document outlines the safety procedures...

🔍 Simple Audit Demo (No LLM Required)
----------------------------------------
DOCUMENT AUDIT REPORT
Document: lab_safety_manual.md
Score: 39.0%
Findings: 5

⚠️ Finding 1: missing_prerequisite
   Line 1: ...
   Question: What are the specific requirements mentioned?
   Recommendation: Add prerequisite information

🎭 Mock Audit Demo
----------------------------------------
Score: 24.0%
Findings: 6
- ⚠️ Missing: CFM rate for ventilation
- 📝 Ambiguous: What PPE required for acids?
- ⚠️ Missing: Emergency phone number
```

---

## Project Structure

```
agent-reliability-harness/
├── arh/                          # Main package
│   ├── core/                     # Core components (Step 1-2)
│   │   ├── models.py             # Data structures
│   │   ├── agent_wrapper.py      # LLM API wrappers
│   │   └── harness.py            # Test orchestrator
│   ├── tests/                    # Reliability tests (Step 2)
│   │   ├── robustness.py         # Perturbation testing
│   │   ├── consistency.py        # Variance testing
│   │   ├── groundedness.py       # Hallucination detection
│   │   └── predictability.py     # Latency testing
│   └── auditor/                  # Adversarial auditor (Step 3)
│       ├── proposer.py           # Question generator
│       ├── solver.py             # Document-only answerer
│       ├── evaluator.py          # Flaw classifier
│       └── auditor.py            # Main orchestrator
├── examples/                     # Demo scripts
│   ├── test_wrapper.py           # Step 1 demo
│   ├── test_harness.py           # Step 2 demo
│   └── test_auditor.py           # Step 3 demo
└── docs/                         # Documentation
```

---

## How Each Step Works

### Step 1: Agent Wrapper (Foundation)

```
┌──────────────┐      ┌──────────────────┐      ┌──────────────┐
│  Your Prompt │─────▶│   AgentWrapper   │─────▶│   LLM API    │
└──────────────┘      │ (OpenAI/Ollama)  │      │ (or Mock)    │
                      └──────────────────┘      └──────────────┘
                               │
                               ▼
                      ┌──────────────────┐
                      │  AgentResponse   │
                      │ - content        │
                      │ - latency_ms     │
                      │ - model          │
                      └──────────────────┘
```

**Key Files:**
- `models.py` - Defines `AgentResponse`, `TestResult`, `Finding`
- `agent_wrapper.py` - Unified interface for all LLM providers

---

### Step 2: Reliability Harness

```
                      ┌────────────────────────────────────────────┐
                      │            ReliabilityHarness              │
                      ├────────────────────────────────────────────┤
 Your Prompts ──────▶ │  ┌────────────┐  ┌────────────┐           │
                      │  │ Robustness │  │Consistency │           │
                      │  │   Test     │  │   Test     │           │
                      │  └────────────┘  └────────────┘           │
                      │  ┌────────────┐  ┌────────────┐           │
                      │  │Groundedness│  │Predictabil-│           │
                      │  │   Test     │  │ity Test    │           │
                      │  └────────────┘  └────────────┘           │
                      └────────────────────────────────────────────┘
                                         │
                                         ▼
                      ┌────────────────────────────────────────────┐
                      │              Final Report                   │
                      │  Score: 96.7%  │  Verdict: PASS            │
                      └────────────────────────────────────────────┘
```

**What Each Test Does:**
| Test | What It Measures |
|------|------------------|
| **Robustness** | Does agent handle typos, rephrasing, noise? |
| **Consistency** | Same question 5x = same answer? |
| **Groundedness** | Does agent admit uncertainty? |
| **Predictability** | P50/P95/P99 latency distribution |

---

### Step 3: Adversarial Auditor

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│    PROPOSER    │────▶│     SOLVER     │────▶│   EVALUATOR    │
│ (Generate Qs)  │     │ (Answer from   │     │ (Classify      │
│                │     │  doc ONLY)     │     │  failures)     │
└────────────────┘     └────────────────┘     └────────────────┘
        │                      │                      │
        ▼                      ▼                      ▼
 "What's the            STATUS: NOT_FOUND        FLAW: MISSING_PREREQ
  CFM rate?"            CONFIDENCE: 10%          SEVERITY: HIGH
```

**Flaw Types Detected:**
| Flaw | Severity | Meaning |
|------|----------|---------|
| `SAFETY_GAP` | Critical | Missing safety info |
| `MISSING_PREREQ` | High | Missing prerequisites |
| `AMBIGUOUS` | Medium | Unclear language |
| `IMPLICIT_ASSUMPTION` | Medium | Unstated assumptions |
| `TEMPORAL_GAP` | Low | Missing sequence info |

---

## Testing with Real APIs

### OpenAI

```bash
export OPENAI_API_KEY="sk-your-key-here"
python3 examples/test_harness.py
python3 examples/test_auditor.py
```

### Ollama (Local)

```bash
ollama serve
python3 examples/test_wrapper.py
```

---

## Understanding Scores

### Reliability Scores (Step 2)
| Score | Verdict | Meaning |
|-------|---------|---------|
| **85%+** | PASS | Safe to deploy |
| **70-84%** | CONDITIONAL_PASS | Review needed |
| **<70%** | BLOCK | Not ready |

### Audit Scores (Step 3)
| Score | Meaning |
|-------|---------|
| **90%+** | Excellent documentation |
| **70-89%** | Good, minor gaps |
| **50-69%** | Needs improvement |
| **<50%** | Major flaws detected |

---

## Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip3 install -r requirements.txt` |
| `OPENAI_API_KEY not set` | Set via `export OPENAI_API_KEY="your-key"` |
| Ollama not available | Start with `ollama serve` |

---

## Next Steps

- **Step 4**: CLI interface (`arh test --agent URL`)
- **Step 5**: Metrics export (Prometheus/Grafana)
