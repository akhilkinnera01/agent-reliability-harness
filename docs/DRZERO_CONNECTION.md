# Research Lineage: From Dr. Zero to ARH

## Overview

The Agent Reliability Harness (ARH) Adversarial Auditor module is directly inspired by the proposer-solver framework introduced in Meta's Dr. Zero paper. This document explains the connection and our novel adaptation.

---

## The Original Dr. Zero Insight

The Dr. Zero paper (Yue et al., Meta, January 2025) introduces a self-evolving framework for training search agents:

### Key Components

1. **Proposer**: Generates increasingly difficult questions
2. **Solver**: Attempts to answer them using search
3. **The Critical Signal**: **Partial failure** indicates the most valuable learning opportunity

### The Core Insight

> "When the solver partially fails—finding some relevant information but not enough to fully answer the question—that's the most valuable training signal."

This is powerful because:
- **Complete success** = Too easy, not useful for learning
- **Complete failure** = Too hard, no gradient to learn from
- **Partial failure** = The "goldilocks zone" for improvement

---

## Our Adaptation: Documentation Quality Assurance

We realized this framework isn't just for training AI models—it's a general pattern for **finding knowledge gaps**.

### The Reframe

| Dr. Zero | ARH Adversarial Auditor |
|----------|-------------------------|
| Proposer generates questions | Proposer generates adversarial questions |
| Solver uses search engine | Solver uses ONLY the document |
| Failure → training signal | Failure → documentation flaw |
| Goal: improve model | Goal: improve documentation |

### The Key Question

> "If an AI constrained to the document can't answer a reasonable question, a human will get it wrong too."

This transforms AI from a passive answerer into an **active auditor**.

---

## How It Works in ARH

### 1. Proposer Module

Generates adversarial questions with varying "hop complexity":

| Hop Level | Type | Example |
|-----------|------|---------|
| 1 | Direct fact | "What is the timeout value?" |
| 2 | Cross-reference | "Does step 3 depend on step 1?" |
| 3 | Multi-section synthesis | "What's the full deployment flow?" |
| 4 | Edge case reasoning | "What if the server is down?" |

### 2. Solver Module

Attempts to answer using **ONLY** the document:
- Cannot use external knowledge
- Must cite specific text
- Reports confidence and status (FOUND/NOT_FOUND/AMBIGUOUS/PARTIAL)

### 3. Evaluator Module

Classifies failures into actionable flaw types:
- `SAFETY_GAP` - Missing safety information
- `MISSING_PREREQ` - Missing prerequisites
- `AMBIGUOUS` - Unclear language
- `IMPLICIT_ASSUMPTION` - Unstated assumptions

---

## Why This Matters

### For Documentation

Every finding is **human-verifiable in seconds**:
- See the question that couldn't be answered
- See the exact gap in the document
- Get specific recommendations for fixing it

### For AI Systems

The output isn't "your system is 73% good"—it's **specific failures with specific fixes**.

---

## The Interview Story

> "I built the Agent Reliability Harness after reading Meta's Dr. Zero paper on self-evolving search agents. I realized the proposer-solver pattern isn't just for training—it's a general framework for finding knowledge gaps.
>
> ARH applies SRE principles to AI: robustness testing, consistency checks, hallucination detection, and latency profiling. The Adversarial Auditor module turns AI into a documentation quality tool—it generates questions that should be answerable from your docs, and when the solver fails, you've found a gap.
>
> Every finding is human-verifiable in seconds. The output isn't 'your system is 73% good'—it's specific failures with specific fixes."

---

## References

1. Yue et al. "Dr. Zero: Self-Evolving Search Agents without Training Data" (Meta, January 2025)
2. SRE Book - Site Reliability Engineering (Google)
3. Agent Testing Best Practices (Various industry sources)
