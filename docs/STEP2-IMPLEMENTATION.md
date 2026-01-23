# Step 2: Core Reliability Tests - Documentation

**Date**: January 29, 2025  
**Status**: ✅ Complete

---

## Overview

This document describes the implementation of Step 2 (Core Reliability Tests) from the ARH Execution Plan. The goal was to implement the four core reliability dimensions and a test runner harness.

---

## What Was Implemented

### 1. Robustness Test (`arh/tests/robustness.py`)

Tests agent stability under input perturbations:

| Perturbation | Description |
|--------------|-------------|
| `typo` | Swaps adjacent characters in a random word |
| `rephrase` | Adds "Please explain." or "Can you tell me:" |
| `case_shift` | Converts entire prompt to upper/lower case |
| `noise` | Appends random characters to prompt |
| `truncate` | Removes the last word from prompt |

**Scoring**: Consistent responses / Total perturbations

### 2. Consistency Test (`arh/tests/consistency.py`)

Tests response stability across repeated identical queries:

- Queries each prompt multiple times (default: 5)
- Calculates pairwise Jaccard distance between responses
- Score = 1 - average variance

### 3. Groundedness Test (`arh/tests/groundedness.py`)

Detects potential hallucinations using confidence calibration:

- Asks agent for confidence rating on its responses
- Looks for uncertainty signals ("I'm not certain", "I cannot verify", etc.)
- Checks for low stated confidence (<50%)

### 4. Predictability Test (`arh/tests/predictability.py`)

Measures latency distribution and tail behavior:

| Metric | Description |
|--------|-------------|
| P50 | Median latency |
| P95 | 95th percentile latency |
| P99 | 99th percentile latency |
| Variance | Standard deviation of latencies |
| Timeout rate | % of requests exceeding SLO |

### 5. Reliability Harness (`arh/core/harness.py`)

Main orchestrator providing:

- `run_all(prompts)` - Run all 4 tests
- `run_test(name, prompts)` - Run individual test
- `get_overall_score()` - Weighted score calculation
- `get_verdict()` - PASS/CONDITIONAL_PASS/BLOCK
- `generate_report()` - Full JSON report
- `print_summary()` - Human-readable output

**Score Weights**:
- Groundedness: 30%
- Robustness: 25%
- Consistency: 25%
- Predictability: 20%

---

## Files Created

| File | Description |
|------|-------------|
| `arh/tests/robustness.py` | Robustness test class |
| `arh/tests/consistency.py` | Consistency test class |
| `arh/tests/groundedness.py` | Groundedness test class |
| `arh/tests/predictability.py` | Predictability test class |
| `arh/core/harness.py` | ReliabilityHarness orchestrator |
| `examples/test_harness.py` | Demo script with mock agent |

---

## Verification Results

### Syntax Validation
```
✅ robustness.py OK
✅ consistency.py OK
✅ groundedness.py OK
✅ predictability.py OK
✅ harness.py OK
```

### Import Tests
```
✅ All tests import OK
✅ Harness import OK
```

### End-to-End Test Output
```
RELIABILITY ASSESSMENT SUMMARY
Agent: mock-agent
Overall Score: 96.7%
Verdict: PASS
------------------------------------------------------------
✅ Robustness: 86.7%
✅ Consistency: 100.0%
✅ Groundedness: 100.0%
✅ Predictability: 100.0%
```

---

## Issues Encountered

### 1. Numpy Dependency
**Issue**: `numpy` was not installed (required for percentile calculations)

**Resolution**: Installed via `pip3 install numpy`

### 2. None - Implementation was smooth
The execution plan was well-defined and the implementation followed it directly without significant issues.

---

## Changes from Execution Plan

| Change | Reason |
|--------|--------|
| Added `print_summary()` method | Convenience for human-readable output |
| Added `clear_results()` method | Allow re-running tests without creating new harness |
| Enhanced error handling | Skip failed queries gracefully in tests |
| Float conversion for numpy results | Ensure JSON serialization works |

---

## Exit Criteria Status

| Criteria | Status |
|----------|--------|
| Can run all 4 tests on any agent | ✅ Complete |
| Each test produces score + failures + recommendations | ✅ Complete |
| Harness calculates overall score and verdict | ✅ Complete |
| Can run individual tests or full suite | ✅ Complete |

---

## Usage Example

```python
from arh.core.agent_wrapper import OpenAIWrapper
from arh.core.harness import ReliabilityHarness

# Create agent and harness
agent = OpenAIWrapper(api_key="your-key")
harness = ReliabilityHarness(agent)

# Run all tests
prompts = ["What is 2 + 2?", "What is the capital of France?"]
results = harness.run_all(prompts)

# Get overall assessment
print(f"Score: {harness.get_overall_score():.1%}")
print(f"Verdict: {harness.get_verdict()}")

# Generate full report
report = harness.generate_report()
```

---

## Next Steps (Step 3)

The core reliability tests are complete. Step 3 involves implementing the Adversarial Auditor Module:
- Proposer Module (generates adversarial questions)
- Solver Module (answers from document only)
- Evaluator Module (classifies flaw types)
