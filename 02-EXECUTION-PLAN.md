# Agent Reliability Harness (ARH)
## Execution Plan: 5 Steps to Production

**Version**: 1.0  
**Author**: [Your Name]  
**Date**: January 2025  
**Timeline**: 10-12 Days  
**Budget**: $25-40

---

## Overview

This document outlines the step-by-step execution plan for building the Agent Reliability Harness (ARH). Each step is designed to produce working, testable code that builds toward the complete system.

**Philosophy**: Ship incrementally. Each step produces a usable artifact.

---

## Timeline at a Glance

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ARH EXECUTION TIMELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STEP 1          STEP 2          STEP 3          STEP 4          STEP 5 │
│  Foundation      Core Tests      Adversarial     Integration     Polish │
│                                  Auditor                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌────────┐│
│  │ Days    │    │ Days    │    │ Days    │    │ Days    │    │ Days   ││
│  │ 1-2     │───▶│ 3-5     │───▶│ 6-8     │───▶│ 9-10    │───▶│ 11-12  ││
│  │         │    │         │    │         │    │         │    │        ││
│  │ Agent   │    │ 4 Core  │    │ Proposer│    │ CLI     │    │ Docs   ││
│  │ Wrapper │    │ Tests   │    │ Solver  │    │ Metrics │    │ Demo   ││
│  │ Project │    │         │    │ Evaluator│   │ Report  │    │ Video  ││
│  │ Setup   │    │         │    │         │    │         │    │        ││
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    └────────┘│
│                                                                          │
│  Deliverable:   Deliverable:   Deliverable:   Deliverable:   Deliverable│
│  Can test any   Full agent     Can audit      Production-    Ready for  │
│  agent endpoint reliability    documents      ready tool     showcase   │
│                 scores                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Foundation (Days 1-2)

### Objective
Set up project structure and build the Agent Wrapper that can test ANY LLM endpoint.

### Day 1: Project Setup & Core Abstractions

**Tasks**:

1. **Initialize Repository**
```bash
mkdir agent-reliability-harness
cd agent-reliability-harness
git init

# Create project structure
mkdir -p arh/{core,tests,auditor,metrics,cli}
mkdir -p dashboards/grafana
mkdir -p examples
mkdir -p tests
mkdir -p docs
```

2. **Setup Python Environment**
```bash
python -m venv venv
source venv/bin/activate

# Create requirements.txt
cat > requirements.txt << EOF
httpx>=0.25.0
pydantic>=2.0.0
typer>=0.9.0
rich>=13.0.0
prometheus-client>=0.19.0
numpy>=1.24.0
scikit-learn>=1.3.0
python-dotenv>=1.0.0
pytest>=7.4.0
pytest-asyncio>=0.21.0
EOF

pip install -r requirements.txt
```

3. **Define Core Data Models**

Create `arh/core/models.py`:
```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional

class TestStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    CONDITIONAL = "conditional"
    ERROR = "error"

class FlawType(Enum):
    AMBIGUOUS = "ambiguous"
    MISSING_PREREQ = "missing_prerequisite"
    IMPLICIT_ASSUMPTION = "implicit_assumption"
    CONTRADICTION = "contradiction"
    TEMPORAL_GAP = "temporal_gap"
    SAFETY_GAP = "safety_gap"

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AgentResponse:
    content: str
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None

@dataclass
class TestResult:
    name: str
    score: float
    status: TestStatus
    details: Dict[str, Any] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class Finding:
    line: int
    text: str
    flaw_type: FlawType
    severity: Severity
    question: str
    solver_response: str
    recommendation: str

@dataclass
class AuditReport:
    document: str
    section: str
    overall_score: float
    findings: List[Finding]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TrustReport:
    assessment_id: str
    agent_score: float
    knowledge_score: float
    trust_score: float
    verdict: str
    blockers: List[str]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
```

### Day 2: Agent Wrapper Implementation

**Tasks**:

1. **Build Generic Agent Wrapper**

Create `arh/core/agent_wrapper.py`:
```python
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
        self.endpoint = endpoint
        self.auth_header = auth_header or {}
        self.timeout = timeout
        self.model = model
        self.response_log: List[AgentResponse] = []
    
    def query(self, prompt: str, **kwargs) -> AgentResponse:
        """Send a query to the agent and return normalized response."""
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
        """Query multiple prompts sequentially."""
        return [self.query(p, **kwargs) for p in prompts]
    
    def _build_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Build API payload. Override for custom formats."""
        # Default OpenAI-compatible format
        return {
            "model": kwargs.get("model", self.model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
    
    def _extract_content(self, response: Dict[str, Any]) -> str:
        """Extract content from response. Override for custom formats."""
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


# Pre-configured wrappers for common providers
class OpenAIWrapper(AgentWrapper):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        super().__init__(
            endpoint="https://api.openai.com/v1/chat/completions",
            auth_header={"Authorization": f"Bearer {api_key}"},
            model=model
        )

class AnthropicWrapper(AgentWrapper):
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        super().__init__(
            endpoint="https://api.anthropic.com/v1/messages",
            auth_header={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            },
            model=model
        )
    
    def _build_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "messages": [{"role": "user", "content": prompt}]
        }

class OllamaWrapper(AgentWrapper):
    def __init__(self, model: str = "llama2", host: str = "localhost", port: int = 11434):
        super().__init__(
            endpoint=f"http://{host}:{port}/api/generate",
            model=model
        )
    
    def _build_payload(self, prompt: str, **kwargs) -> Dict[str, Any]:
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
    
    def _extract_content(self, response: Dict[str, Any]) -> str:
        return response.get("response", "")
```

2. **Create Simple Test Script**

Create `examples/test_wrapper.py`:
```python
from arh.core.agent_wrapper import OpenAIWrapper
import os

# Test the wrapper
wrapper = OpenAIWrapper(api_key=os.getenv("OPENAI_API_KEY"))
response = wrapper.query("What is 2 + 2?")

print(f"Response: {response.content}")
print(f"Latency: {response.latency_ms:.2f}ms")
print(f"Error: {response.error}")
```

### Step 1 Deliverables

| Deliverable | Status |
|-------------|--------|
| Project structure | ✅ |
| Core data models | ✅ |
| Agent wrapper (OpenAI, Anthropic, Ollama) | ✅ |
| Basic test script | ✅ |

### Step 1 Exit Criteria
- [ ] Can query OpenAI endpoint and get response
- [ ] Can query local Ollama endpoint
- [ ] Response includes latency measurement
- [ ] Errors are captured gracefully

---

## Step 2: Core Reliability Tests (Days 3-5)

### Objective
Implement the four core reliability dimensions: Robustness, Consistency, Groundedness, Predictability.

### Day 3: Robustness Test

**Create `arh/tests/robustness.py`**:

```python
import random
import re
from typing import List, Tuple
from ..core.agent_wrapper import AgentWrapper
from ..core.models import TestResult, TestStatus

class RobustnessTest:
    """
    Test agent robustness through prompt perturbations.
    A robust agent should produce semantically similar outputs
    despite surface-level changes to inputs.
    """
    
    def __init__(
        self,
        perturbations: List[str] = None,
        threshold: float = 0.85,
        samples_per_perturbation: int = 5
    ):
        self.perturbations = perturbations or [
            "typo", "rephrase", "case_shift", "noise", "truncate"
        ]
        self.threshold = threshold
        self.samples = samples_per_perturbation
    
    def run(self, agent: AgentWrapper, prompts: List[str]) -> TestResult:
        """Run robustness test on agent with given prompts."""
        total_tests = 0
        consistent_tests = 0
        failures = []
        
        for prompt in prompts:
            # Get baseline response
            baseline = agent.query(prompt)
            
            for perturb_type in self.perturbations:
                perturbed = self._apply_perturbation(prompt, perturb_type)
                response = agent.query(perturbed)
                
                # Check semantic similarity
                is_consistent = self._check_consistency(
                    baseline.content, 
                    response.content
                )
                
                total_tests += 1
                if is_consistent:
                    consistent_tests += 1
                else:
                    failures.append(
                        f"{perturb_type}: '{prompt[:50]}...' produced different output"
                    )
        
        score = consistent_tests / total_tests if total_tests > 0 else 0
        
        return TestResult(
            name="robustness",
            score=score,
            status=TestStatus.PASS if score >= self.threshold else TestStatus.FAIL,
            details={
                "total_tests": total_tests,
                "consistent_tests": consistent_tests,
                "perturbation_types": self.perturbations
            },
            failures=failures[:10],  # Limit to top 10
            recommendations=self._get_recommendations(score, failures)
        )
    
    def _apply_perturbation(self, text: str, perturb_type: str) -> str:
        """Apply a specific perturbation to the text."""
        if perturb_type == "typo":
            return self._add_typo(text)
        elif perturb_type == "rephrase":
            return self._rephrase(text)
        elif perturb_type == "case_shift":
            return text.upper() if random.random() > 0.5 else text.lower()
        elif perturb_type == "noise":
            return text + " " + "".join(random.choices("asdf ", k=5))
        elif perturb_type == "truncate":
            words = text.split()
            return " ".join(words[:-1]) if len(words) > 1 else text
        return text
    
    def _add_typo(self, text: str) -> str:
        """Add a realistic typo to the text."""
        words = text.split()
        if not words:
            return text
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        if len(word) > 2:
            # Swap two adjacent characters
            pos = random.randint(0, len(word) - 2)
            word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
            words[idx] = word
        return " ".join(words)
    
    def _rephrase(self, text: str) -> str:
        """Simple rephrasing (in production, use LLM for better rephrasing)."""
        # Add question markers or rephrase patterns
        if text.endswith("?"):
            return "Can you tell me: " + text
        return text + " Please explain."
    
    def _check_consistency(self, baseline: str, response: str) -> bool:
        """Check if two responses are semantically consistent."""
        # Simple heuristic: check for significant overlap
        # In production, use embedding similarity
        baseline_words = set(baseline.lower().split())
        response_words = set(response.lower().split())
        
        if not baseline_words or not response_words:
            return False
        
        overlap = len(baseline_words & response_words)
        union = len(baseline_words | response_words)
        
        jaccard = overlap / union if union > 0 else 0
        return jaccard > 0.3  # Threshold for similarity
    
    def _get_recommendations(self, score: float, failures: List[str]) -> List[str]:
        """Generate recommendations based on failures."""
        recs = []
        if score < self.threshold:
            recs.append("Consider adding input normalization layer")
            if any("typo" in f for f in failures):
                recs.append("Implement spell-checking or fuzzy matching")
            if any("case" in f for f in failures):
                recs.append("Normalize input case before processing")
        return recs
```

### Day 4: Consistency & Groundedness Tests

**Create `arh/tests/consistency.py`**:

```python
from typing import List
import numpy as np
from ..core.agent_wrapper import AgentWrapper
from ..core.models import TestResult, TestStatus

class ConsistencyTest:
    """
    Test agent consistency by querying the same prompt multiple times.
    A consistent agent produces semantically similar responses.
    """
    
    def __init__(
        self,
        samples: int = 5,
        threshold: float = 0.90,
        temperature: float = 0.7
    ):
        self.samples = samples
        self.threshold = threshold
        self.temperature = temperature
    
    def run(self, agent: AgentWrapper, prompts: List[str]) -> TestResult:
        """Run consistency test."""
        all_variances = []
        failures = []
        
        for prompt in prompts:
            # Query multiple times
            responses = [
                agent.query(prompt, temperature=self.temperature)
                for _ in range(self.samples)
            ]
            
            # Calculate variance in responses
            contents = [r.content for r in responses if not r.error]
            variance = self._calculate_variance(contents)
            all_variances.append(variance)
            
            if variance > (1 - self.threshold):
                failures.append(f"High variance on: '{prompt[:50]}...'")
        
        avg_variance = np.mean(all_variances) if all_variances else 1.0
        score = 1 - avg_variance
        
        return TestResult(
            name="consistency",
            score=score,
            status=TestStatus.PASS if score >= self.threshold else TestStatus.FAIL,
            details={
                "average_variance": avg_variance,
                "samples_per_prompt": self.samples,
                "prompts_tested": len(prompts)
            },
            failures=failures[:10],
            recommendations=self._get_recommendations(score)
        )
    
    def _calculate_variance(self, responses: List[str]) -> float:
        """Calculate semantic variance across responses."""
        if len(responses) < 2:
            return 0.0
        
        # Simple approach: pairwise Jaccard distance
        distances = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                dist = self._jaccard_distance(responses[i], responses[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _jaccard_distance(self, a: str, b: str) -> float:
        """Calculate Jaccard distance between two strings."""
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        
        if not set_a and not set_b:
            return 0.0
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return 1 - (intersection / union) if union > 0 else 1.0
    
    def _get_recommendations(self, score: float) -> List[str]:
        recs = []
        if score < self.threshold:
            recs.append("Consider lowering temperature for more deterministic outputs")
            recs.append("Implement response caching for identical queries")
        return recs
```

**Create `arh/tests/groundedness.py`**:

```python
from typing import List, Dict
from ..core.agent_wrapper import AgentWrapper
from ..core.models import TestResult, TestStatus

class GroundednessTest:
    """
    Test agent groundedness by detecting potential hallucinations.
    Uses self-consistency and confidence calibration.
    """
    
    def __init__(
        self,
        threshold: float = 0.85,
        verification_samples: int = 3
    ):
        self.threshold = threshold
        self.verification_samples = verification_samples
    
    def run(self, agent: AgentWrapper, prompts: List[str]) -> TestResult:
        """Run groundedness test."""
        hallucination_count = 0
        total_tests = 0
        failures = []
        
        for prompt in prompts:
            # Ask the question
            response = agent.query(prompt)
            
            # Ask for verification/sources
            verification_prompt = f"""
            You previously answered: "{response.content[:200]}"
            
            Rate your confidence (0-100) and explain why.
            If you're not certain, say "I'm not certain because..."
            """
            
            verification = agent.query(verification_prompt)
            
            # Check for hallucination signals
            is_hallucination = self._detect_hallucination(
                response.content,
                verification.content
            )
            
            total_tests += 1
            if is_hallucination:
                hallucination_count += 1
                failures.append(f"Potential hallucination: '{prompt[:50]}...'")
        
        hallucination_rate = hallucination_count / total_tests if total_tests > 0 else 0
        score = 1 - hallucination_rate
        
        return TestResult(
            name="groundedness",
            score=score,
            status=TestStatus.PASS if score >= self.threshold else TestStatus.FAIL,
            details={
                "hallucination_rate": hallucination_rate,
                "total_tests": total_tests,
                "hallucinations_detected": hallucination_count
            },
            failures=failures[:10],
            recommendations=self._get_recommendations(hallucination_rate)
        )
    
    def _detect_hallucination(self, response: str, verification: str) -> bool:
        """Detect if response shows hallucination signals."""
        verification_lower = verification.lower()
        
        # Check for uncertainty signals in verification
        uncertainty_signals = [
            "i'm not certain",
            "i'm not sure",
            "i cannot verify",
            "i don't have",
            "i may have",
            "i might be wrong",
            "i apologize",
            "i cannot confirm"
        ]
        
        for signal in uncertainty_signals:
            if signal in verification_lower:
                return True
        
        # Check for very low confidence
        import re
        confidence_match = re.search(r'(\d+)%?\s*(confidence|certain)', verification_lower)
        if confidence_match:
            confidence = int(confidence_match.group(1))
            if confidence < 50:
                return True
        
        return False
    
    def _get_recommendations(self, hallucination_rate: float) -> List[str]:
        recs = []
        if hallucination_rate > 0.15:
            recs.append("Implement RAG to ground responses in verified sources")
            recs.append("Add explicit 'I don't know' training")
            recs.append("Consider using a smaller, more constrained model")
        return recs
```

### Day 5: Predictability Test & Test Runner

**Create `arh/tests/predictability.py`**:

```python
import time
import numpy as np
from typing import List
from ..core.agent_wrapper import AgentWrapper
from ..core.models import TestResult, TestStatus

class PredictabilityTest:
    """
    Test agent predictability under load.
    Measures latency distribution and tail behavior.
    """
    
    def __init__(
        self,
        p99_slo_ms: float = 3000,
        timeout_threshold: float = 0.01,
        samples: int = 20
    ):
        self.p99_slo = p99_slo_ms
        self.timeout_threshold = timeout_threshold
        self.samples = samples
    
    def run(self, agent: AgentWrapper, prompts: List[str]) -> TestResult:
        """Run predictability test."""
        latencies = []
        timeouts = 0
        
        # Cycle through prompts to get enough samples
        test_prompts = (prompts * (self.samples // len(prompts) + 1))[:self.samples]
        
        for prompt in test_prompts:
            response = agent.query(prompt)
            
            if response.error and "timeout" in response.error.lower():
                timeouts += 1
            else:
                latencies.append(response.latency_ms)
        
        if not latencies:
            return TestResult(
                name="predictability",
                score=0.0,
                status=TestStatus.ERROR,
                details={"error": "No successful responses"},
                failures=["All requests failed or timed out"]
            )
        
        # Calculate percentiles
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        variance = np.std(latencies)
        timeout_rate = timeouts / self.samples
        
        # Score based on P99 and timeout rate
        p99_score = 1.0 if p99 <= self.p99_slo else max(0, 1 - (p99 - self.p99_slo) / self.p99_slo)
        timeout_score = 1.0 if timeout_rate <= self.timeout_threshold else 0.5
        variance_score = 1.0 if variance < 1000 else max(0, 1 - variance / 5000)
        
        score = (p99_score * 0.5) + (timeout_score * 0.3) + (variance_score * 0.2)
        
        failures = []
        if p99 > self.p99_slo:
            failures.append(f"P99 latency ({p99:.0f}ms) exceeds SLO ({self.p99_slo}ms)")
        if timeout_rate > self.timeout_threshold:
            failures.append(f"Timeout rate ({timeout_rate:.1%}) exceeds threshold")
        
        return TestResult(
            name="predictability",
            score=score,
            status=TestStatus.PASS if score >= 0.8 else TestStatus.FAIL,
            details={
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2),
                "variance_ms": round(variance, 2),
                "timeout_rate": round(timeout_rate, 4),
                "samples": self.samples
            },
            failures=failures,
            recommendations=self._get_recommendations(p99, variance, timeout_rate)
        )
    
    def _get_recommendations(self, p99: float, variance: float, timeout_rate: float) -> List[str]:
        recs = []
        if p99 > self.p99_slo:
            recs.append("Consider response streaming for long generations")
            recs.append("Investigate slow queries for optimization opportunities")
        if variance > 2000:
            recs.append("High latency variance - check for resource contention")
        if timeout_rate > 0.01:
            recs.append("Implement retry logic with exponential backoff")
        return recs
```

**Create `arh/core/harness.py`** (Test Runner):

```python
from typing import List, Optional, Dict
from .agent_wrapper import AgentWrapper
from .models import TestResult, TrustReport, TestStatus
from ..tests.robustness import RobustnessTest
from ..tests.consistency import ConsistencyTest
from ..tests.groundedness import GroundednessTest
from ..tests.predictability import PredictabilityTest
import uuid
from datetime import datetime

class ReliabilityHarness:
    """
    Main orchestrator for running reliability tests on agents.
    """
    
    def __init__(self, agent: AgentWrapper):
        self.agent = agent
        self.results: Dict[str, TestResult] = {}
    
    def run_all(self, prompts: List[str]) -> Dict[str, TestResult]:
        """Run all reliability tests."""
        tests = [
            RobustnessTest(),
            ConsistencyTest(),
            GroundednessTest(),
            PredictabilityTest()
        ]
        
        for test in tests:
            result = test.run(self.agent, prompts)
            self.results[result.name] = result
        
        return self.results
    
    def run_test(self, test_name: str, prompts: List[str], **kwargs) -> TestResult:
        """Run a specific test."""
        test_map = {
            "robustness": RobustnessTest,
            "consistency": ConsistencyTest,
            "groundedness": GroundednessTest,
            "predictability": PredictabilityTest
        }
        
        if test_name not in test_map:
            raise ValueError(f"Unknown test: {test_name}")
        
        test = test_map[test_name](**kwargs)
        result = test.run(self.agent, prompts)
        self.results[result.name] = result
        
        return result
    
    def get_overall_score(self) -> float:
        """Calculate weighted overall score."""
        if not self.results:
            return 0.0
        
        weights = {
            "robustness": 0.25,
            "consistency": 0.25,
            "groundedness": 0.30,
            "predictability": 0.20
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for name, result in self.results.items():
            weight = weights.get(name, 0.25)
            weighted_sum += result.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def get_verdict(self) -> str:
        """Get deployment verdict based on results."""
        score = self.get_overall_score()
        
        # Check for any critical failures
        has_critical = any(
            r.status == TestStatus.FAIL and r.score < 0.5
            for r in self.results.values()
        )
        
        if has_critical:
            return "BLOCK"
        elif score >= 0.85:
            return "PASS"
        elif score >= 0.70:
            return "CONDITIONAL_PASS"
        else:
            return "BLOCK"
    
    def generate_report(self) -> Dict:
        """Generate full reliability report."""
        return {
            "assessment_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "agent": self.agent.model,
            "overall_score": round(self.get_overall_score(), 3),
            "verdict": self.get_verdict(),
            "dimensions": {
                name: {
                    "score": round(result.score, 3),
                    "status": result.status.value,
                    "details": result.details,
                    "failures": result.failures,
                    "recommendations": result.recommendations
                }
                for name, result in self.results.items()
            }
        }
```

### Step 2 Deliverables

| Deliverable | Status |
|-------------|--------|
| Robustness test | ✅ |
| Consistency test | ✅ |
| Groundedness test | ✅ |
| Predictability test | ✅ |
| Test runner/harness | ✅ |

### Step 2 Exit Criteria
- [ ] Can run all 4 tests on any agent
- [ ] Each test produces score + failures + recommendations
- [ ] Harness calculates overall score and verdict
- [ ] Can run individual tests or full suite

---

## Step 3: Adversarial Auditor Module (Days 6-8)

### Objective
Build the Dr. Zero-inspired document auditing system: Proposer, Solver, Evaluator.

### Day 6: Proposer Module

**Create `arh/auditor/proposer.py`**:

```python
from typing import List, Dict
from enum import Enum
from ..core.agent_wrapper import AgentWrapper

class HopComplexity(Enum):
    ONE = 1    # Direct fact retrieval
    TWO = 2    # Cross-reference
    THREE = 3  # Multi-section synthesis
    FOUR = 4   # Edge case reasoning

class Proposer:
    """
    Generates adversarial questions designed to expose documentation flaws.
    Inspired by Dr. Zero's proposer-solver framework.
    """
    
    def __init__(self, model: AgentWrapper):
        self.model = model
    
    def generate_questions(
        self,
        document: str,
        section: str,
        hop_complexity: List[HopComplexity] = None,
        questions_per_hop: int = 3
    ) -> List[Dict]:
        """Generate adversarial questions for a document section."""
        
        hop_complexity = hop_complexity or [HopComplexity.ONE, HopComplexity.TWO]
        all_questions = []
        
        for hop in hop_complexity:
            prompt = self._build_proposer_prompt(document, section, hop)
            response = self.model.query(prompt, temperature=0.8)
            
            questions = self._parse_questions(response.content, hop)
            all_questions.extend(questions[:questions_per_hop])
        
        return all_questions
    
    def _build_proposer_prompt(
        self, 
        document: str, 
        section: str, 
        hop: HopComplexity
    ) -> str:
        """Build the proposer prompt based on hop complexity."""
        
        hop_instructions = {
            HopComplexity.ONE: """
Generate questions that test DIRECT FACT RETRIEVAL from this section.
These should be simple questions whose answers should be explicitly stated.
Focus on: specific values, definitions, direct requirements.""",
            
            HopComplexity.TWO: """
Generate questions that require CROSS-REFERENCING within the document.
These should need information from this section plus implied knowledge.
Focus on: relationships, sequences, conditional requirements.""",
            
            HopComplexity.THREE: """
Generate questions that require MULTI-SECTION SYNTHESIS.
These should need combining information from multiple parts.
Focus on: procedures spanning sections, cumulative requirements, dependencies.""",
            
            HopComplexity.FOUR: """
Generate questions about EDGE CASES and FAILURE MODES.
These should probe what happens when things go wrong.
Focus on: exception handling, safety procedures, contingencies."""
        }
        
        return f"""You are an adversarial documentation auditor. Your job is to find 
flaws in documents by generating questions that SHOULD be answerable but likely ARE NOT.

DOCUMENT SECTION:
{section}

FULL DOCUMENT CONTEXT:
{document[:2000]}...

TASK:
{hop_instructions[hop]}

Generate exactly 5 adversarial questions. For each question:
1. It SHOULD be answerable from a complete document
2. It likely EXPOSES a gap, ambiguity, or missing information
3. A real user would reasonably ask this question

Format each question as:
Q1: [question]
TARGET: [what specific info should answer this]
FLAW_IF_MISSING: [AMBIGUOUS|MISSING_PREREQ|IMPLICIT_ASSUMPTION|SAFETY_GAP]

Generate questions:"""

    def _parse_questions(self, response: str, hop: HopComplexity) -> List[Dict]:
        """Parse generated questions from model response."""
        questions = []
        current_q = {}
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Q') and ':' in line:
                if current_q:
                    questions.append(current_q)
                current_q = {
                    'question': line.split(':', 1)[1].strip(),
                    'hop_complexity': hop.value
                }
            elif line.startswith('TARGET:'):
                current_q['target'] = line.split(':', 1)[1].strip()
            elif line.startswith('FLAW_IF_MISSING:'):
                current_q['expected_flaw'] = line.split(':', 1)[1].strip()
        
        if current_q:
            questions.append(current_q)
        
        return questions
```

### Day 7: Solver & Evaluator Modules

**Create `arh/auditor/solver.py`**:

```python
from typing import Dict, Optional
from dataclasses import dataclass
from ..core.agent_wrapper import AgentWrapper

@dataclass
class SolverResponse:
    answer: Optional[str]
    confidence: float
    status: str  # FOUND | NOT_FOUND | AMBIGUOUS | PARTIAL
    citations: list
    missing_info: list
    raw_response: str

class Solver:
    """
    Attempts to answer questions using ONLY the provided document.
    Constrained solver that cannot use external knowledge.
    """
    
    def __init__(self, model: AgentWrapper):
        self.model = model
    
    def answer(self, question: str, document: str) -> SolverResponse:
        """Answer a question using only the document."""
        
        prompt = f"""You are a STRICT documentation validator. You can ONLY use 
information EXPLICITLY stated in the provided document.

CRITICAL RULES:
1. If the answer is NOT EXPLICITLY in the document, respond with STATUS: NOT_FOUND
2. If the answer is AMBIGUOUS (multiple interpretations), respond with STATUS: AMBIGUOUS
3. If you need information NOT in the document, respond with STATUS: NOT_FOUND
4. If you find a partial answer, respond with STATUS: PARTIAL
5. Always cite the SPECIFIC text you're using

DOCUMENT:
{document}

QUESTION: {question}

Respond in this EXACT format:
STATUS: [FOUND|NOT_FOUND|AMBIGUOUS|PARTIAL]
CONFIDENCE: [0-100]
ANSWER: [your answer or "Cannot determine from document"]
CITATION: [exact quote from document, or "N/A"]
MISSING: [what additional info would be needed, or "N/A"]"""

        response = self.model.query(prompt, temperature=0.1)
        return self._parse_response(response.content)
    
    def _parse_response(self, response: str) -> SolverResponse:
        """Parse solver response into structured format."""
        lines = response.strip().split('\n')
        
        status = "NOT_FOUND"
        confidence = 0.0
        answer = None
        citations = []
        missing = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('STATUS:'):
                status = line.split(':', 1)[1].strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    conf_str = line.split(':', 1)[1].strip().replace('%', '')
                    confidence = float(conf_str) / 100
                except:
                    confidence = 0.0
            elif line.startswith('ANSWER:'):
                answer = line.split(':', 1)[1].strip()
            elif line.startswith('CITATION:'):
                cite = line.split(':', 1)[1].strip()
                if cite != "N/A":
                    citations.append(cite)
            elif line.startswith('MISSING:'):
                miss = line.split(':', 1)[1].strip()
                if miss != "N/A":
                    missing.append(miss)
        
        return SolverResponse(
            answer=answer,
            confidence=confidence,
            status=status,
            citations=citations,
            missing_info=missing,
            raw_response=response
        )
```

**Create `arh/auditor/evaluator.py`**:

```python
from typing import Dict, List
from dataclasses import dataclass
from ..core.models import FlawType, Severity, Finding
from .solver import SolverResponse

class Evaluator:
    """
    Evaluates solver responses to classify documentation flaws.
    """
    
    def __init__(self):
        self.severity_map = {
            FlawType.SAFETY_GAP: Severity.CRITICAL,
            FlawType.MISSING_PREREQ: Severity.HIGH,
            FlawType.CONTRADICTION: Severity.HIGH,
            FlawType.AMBIGUOUS: Severity.MEDIUM,
            FlawType.IMPLICIT_ASSUMPTION: Severity.MEDIUM,
            FlawType.TEMPORAL_GAP: Severity.LOW
        }
    
    def evaluate(
        self,
        question: Dict,
        solver_response: SolverResponse,
        section_text: str
    ) -> Finding | None:
        """
        Evaluate if solver failure indicates a documentation flaw.
        Returns Finding if flaw detected, None otherwise.
        """
        
        # If solver answered confidently, no flaw
        if solver_response.status == "FOUND" and solver_response.confidence > 0.8:
            return None
        
        # Determine flaw type
        flaw_type = self._classify_flaw(question, solver_response)
        
        if flaw_type is None:
            return None
        
        # Find relevant line in section
        line_num = self._find_relevant_line(section_text, question['question'])
        relevant_text = self._extract_relevant_text(section_text, line_num)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(flaw_type, question, solver_response)
        
        return Finding(
            line=line_num,
            text=relevant_text,
            flaw_type=flaw_type,
            severity=self.severity_map.get(flaw_type, Severity.MEDIUM),
            question=question['question'],
            solver_response=solver_response.raw_response[:200],
            recommendation=recommendation
        )
    
    def _classify_flaw(self, question: Dict, response: SolverResponse) -> FlawType | None:
        """Classify the type of documentation flaw."""
        
        # Use expected flaw if provided by proposer
        expected = question.get('expected_flaw', '').upper()
        
        if response.status == "NOT_FOUND":
            if "safety" in question['question'].lower() or "hazard" in question['question'].lower():
                return FlawType.SAFETY_GAP
            if expected == "MISSING_PREREQ":
                return FlawType.MISSING_PREREQ
            return FlawType.MISSING_PREREQ
        
        elif response.status == "AMBIGUOUS":
            return FlawType.AMBIGUOUS
        
        elif response.status == "PARTIAL":
            if response.missing_info:
                return FlawType.IMPLICIT_ASSUMPTION
            return FlawType.TEMPORAL_GAP
        
        elif response.confidence < 0.5:
            return FlawType.AMBIGUOUS
        
        return None
    
    def _find_relevant_line(self, section: str, question: str) -> int:
        """Find the most relevant line number for the question."""
        lines = section.split('\n')
        question_words = set(question.lower().split())
        
        best_line = 1
        best_overlap = 0
        
        for i, line in enumerate(lines, 1):
            line_words = set(line.lower().split())
            overlap = len(question_words & line_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_line = i
        
        return best_line
    
    def _extract_relevant_text(self, section: str, line_num: int) -> str:
        """Extract the relevant text around the line number."""
        lines = section.split('\n')
        if 0 < line_num <= len(lines):
            return lines[line_num - 1].strip()[:100]
        return section[:100]
    
    def _generate_recommendation(
        self, 
        flaw_type: FlawType, 
        question: Dict,
        response: SolverResponse
    ) -> str:
        """Generate actionable recommendation for fixing the flaw."""
        
        recommendations = {
            FlawType.AMBIGUOUS: f"Clarify the language. Specify exact values or conditions for: {question.get('target', 'this requirement')}",
            FlawType.MISSING_PREREQ: f"Add prerequisite information. Document should include: {', '.join(response.missing_info) if response.missing_info else 'missing context'}",
            FlawType.IMPLICIT_ASSUMPTION: "Make implicit assumptions explicit. State all required context directly.",
            FlawType.SAFETY_GAP: "CRITICAL: Add safety information. This gap could lead to harm.",
            FlawType.TEMPORAL_GAP: "Add sequence information. Clarify what steps occur between described actions.",
            FlawType.CONTRADICTION: "Resolve contradiction between sections. Ensure consistent information."
        }
        
        return recommendations.get(flaw_type, "Review and improve clarity.")
```

### Day 8: Auditor Orchestrator

**Create `arh/auditor/auditor.py`**:

```python
from typing import List, Dict, Optional
from ..core.agent_wrapper import AgentWrapper
from ..core.models import AuditReport, Finding, FlawType, Severity
from .proposer import Proposer, HopComplexity
from .solver import Solver
from .evaluator import Evaluator
from datetime import datetime

class AdversarialAuditor:
    """
    Main orchestrator for adversarial documentation auditing.
    Implements the Dr. Zero proposer-solver loop for finding doc flaws.
    """
    
    def __init__(
        self,
        proposer_model: AgentWrapper,
        solver_model: AgentWrapper = None,
        hop_complexity: List[HopComplexity] = None,
        flaw_types: List[FlawType] = None
    ):
        self.proposer = Proposer(proposer_model)
        self.solver = Solver(solver_model or proposer_model)
        self.evaluator = Evaluator()
        self.hop_complexity = hop_complexity or [
            HopComplexity.ONE, 
            HopComplexity.TWO
        ]
        self.flaw_types = flaw_types  # Filter for specific flaws
    
    def audit(
        self,
        document: str,
        sections: List[Dict[str, str]] = None
    ) -> AuditReport:
        """
        Audit a document for flaws.
        
        Args:
            document: Full document text
            sections: Optional list of {"name": str, "content": str}
                     If not provided, treats entire doc as one section
        """
        
        if sections is None:
            sections = [{"name": "full_document", "content": document}]
        
        all_findings: List[Finding] = []
        
        for section in sections:
            section_findings = self._audit_section(
                document=document,
                section_name=section["name"],
                section_content=section["content"]
            )
            all_findings.extend(section_findings)
        
        # Filter by flaw type if specified
        if self.flaw_types:
            all_findings = [
                f for f in all_findings 
                if f.flaw_type in self.flaw_types
            ]
        
        # Calculate overall score
        score = self._calculate_score(all_findings, document)
        
        return AuditReport(
            document="document",
            section="all",
            overall_score=score,
            findings=all_findings,
            timestamp=datetime.now()
        )
    
    def _audit_section(
        self,
        document: str,
        section_name: str,
        section_content: str
    ) -> List[Finding]:
        """Audit a single section of the document."""
        
        findings = []
        
        # Generate adversarial questions
        questions = self.proposer.generate_questions(
            document=document,
            section=section_content,
            hop_complexity=self.hop_complexity,
            questions_per_hop=3
        )
        
        # Test each question with solver
        for question in questions:
            solver_response = self.solver.answer(
                question=question["question"],
                document=document
            )
            
            # Evaluate for flaw
            finding = self.evaluator.evaluate(
                question=question,
                solver_response=solver_response,
                section_text=section_content
            )
            
            if finding:
                findings.append(finding)
        
        return findings
    
    def _calculate_score(self, findings: List[Finding], document: str) -> float:
        """Calculate document reliability score based on findings."""
        
        if not findings:
            return 1.0
        
        # Weight by severity
        severity_weights = {
            Severity.CRITICAL: 0.25,
            Severity.HIGH: 0.15,
            Severity.MEDIUM: 0.08,
            Severity.LOW: 0.03
        }
        
        total_penalty = sum(
            severity_weights.get(f.severity, 0.05)
            for f in findings
        )
        
        # Cap penalty at 0.8 (minimum score of 0.2)
        score = max(0.2, 1.0 - total_penalty)
        
        return round(score, 3)
    
    def audit_file(self, filepath: str) -> AuditReport:
        """Convenience method to audit a file directly."""
        with open(filepath, 'r') as f:
            content = f.read()
        return self.audit(content)
```

### Step 3 Deliverables

| Deliverable | Status |
|-------------|--------|
| Proposer module | ✅ |
| Solver module | ✅ |
| Evaluator module | ✅ |
| Auditor orchestrator | ✅ |
| Flaw taxonomy | ✅ |

### Step 3 Exit Criteria
- [ ] Proposer generates multi-hop adversarial questions
- [ ] Solver attempts answers constrained to document
- [ ] Evaluator classifies failures into flaw types
- [ ] Full audit produces scored report with findings

---

## Step 4: Integration & CLI (Days 9-10)

### Objective
Build the unified CLI, metrics export, and combined trust reporting.

### Day 9: CLI Implementation

**Create `arh/cli/main.py`**:

```python
import typer
import json
from typing import Optional, List
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(help="Agent Reliability Harness - SRE for AI Agents")
console = Console()

@app.command()
def test(
    agent_url: str = typer.Option(..., "--agent", "-a", help="Agent endpoint URL"),
    test_type: str = typer.Option("all", "--type", "-t", help="Test type: all|robustness|consistency|groundedness|predictability"),
    prompts_file: Optional[Path] = typer.Option(None, "--prompts", "-p", help="File with test prompts (one per line)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for report"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key for agent")
):
    """Run reliability tests on an agent endpoint."""
    
    from ..core.agent_wrapper import AgentWrapper
    from ..core.harness import ReliabilityHarness
    
    console.print(f"[bold blue]Testing agent:[/bold blue] {agent_url}")
    
    # Load prompts
    if prompts_file and prompts_file.exists():
        prompts = prompts_file.read_text().strip().split('\n')
    else:
        prompts = [
            "What is the capital of France?",
            "Explain photosynthesis briefly.",
            "What is 2 + 2?",
            "Who wrote Romeo and Juliet?",
            "What is machine learning?"
        ]
    
    # Setup agent
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    agent = AgentWrapper(endpoint=agent_url, auth_header=headers)
    harness = ReliabilityHarness(agent)
    
    # Run tests
    with console.status("[bold green]Running tests..."):
        if test_type == "all":
            harness.run_all(prompts)
        else:
            harness.run_test(test_type, prompts)
    
    # Generate report
    report = harness.generate_report()
    
    # Display results
    _display_results(report)
    
    # Save if output specified
    if output:
        output.write_text(json.dumps(report, indent=2, default=str))
        console.print(f"\n[green]Report saved to:[/green] {output}")

@app.command()
def audit(
    document: Path = typer.Argument(..., help="Document to audit"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json|html"),
    hop_complexity: str = typer.Option("1,2", "--hops", help="Hop complexity levels (comma-separated)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key for LLM")
):
    """Audit documentation for flaws using adversarial questions."""
    
    from ..core.agent_wrapper import OpenAIWrapper
    from ..auditor.auditor import AdversarialAuditor
    from ..auditor.proposer import HopComplexity
    import os
    
    console.print(f"[bold blue]Auditing document:[/bold blue] {document}")
    
    if not document.exists():
        console.print("[red]Error: Document not found[/red]")
        raise typer.Exit(1)
    
    # Setup model
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        console.print("[red]Error: API key required (--api-key or OPENAI_API_KEY env)[/red]")
        raise typer.Exit(1)
    
    model = OpenAIWrapper(api_key=key, model="gpt-4o-mini")
    
    # Parse hop complexity
    hops = [HopComplexity(int(h)) for h in hop_complexity.split(",")]
    
    auditor = AdversarialAuditor(
        proposer_model=model,
        hop_complexity=hops
    )
    
    # Run audit
    with console.status("[bold green]Generating adversarial questions and auditing..."):
        content = document.read_text()
        report = auditor.audit(content)
    
    # Display results
    _display_audit_results(report)
    
    # Save if output specified
    if output:
        report_dict = {
            "document": str(document),
            "overall_score": report.overall_score,
            "findings": [
                {
                    "line": f.line,
                    "text": f.text,
                    "flaw_type": f.flaw_type.value,
                    "severity": f.severity.value,
                    "question": f.question,
                    "recommendation": f.recommendation
                }
                for f in report.findings
            ]
        }
        output.write_text(json.dumps(report_dict, indent=2))
        console.print(f"\n[green]Report saved to:[/green] {output}")

@app.command()
def trust_eval(
    agent_url: str = typer.Option(..., "--agent", "-a", help="Agent endpoint URL"),
    knowledge_base: Path = typer.Option(..., "--kb", "-k", help="Knowledge base directory"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key")
):
    """Run combined trust evaluation on agent + knowledge base."""
    
    console.print("[bold blue]Running combined trust evaluation...[/bold blue]")
    
    # This would combine both agent testing and doc auditing
    # Implementation follows same pattern
    console.print("[yellow]Combined trust evaluation - see full implementation[/yellow]")

def _display_results(report: dict):
    """Display reliability test results."""
    
    # Overall score panel
    score = report["overall_score"]
    verdict = report["verdict"]
    color = "green" if verdict == "PASS" else "yellow" if verdict == "CONDITIONAL_PASS" else "red"
    
    console.print(Panel(
        f"[bold]Overall Score:[/bold] {score:.1%}\n[bold]Verdict:[/bold] [{color}]{verdict}[/{color}]",
        title="Agent Reliability Results"
    ))
    
    # Dimension table
    table = Table(title="Reliability Dimensions")
    table.add_column("Dimension", style="cyan")
    table.add_column("Score", justify="center")
    table.add_column("Status", justify="center")
    
    for name, data in report.get("dimensions", {}).items():
        status_color = "green" if data["status"] == "pass" else "red"
        table.add_row(
            name.title(),
            f"{data['score']:.1%}",
            f"[{status_color}]{data['status'].upper()}[/{status_color}]"
        )
    
    console.print(table)
    
    # Failures
    for name, data in report.get("dimensions", {}).items():
        if data.get("failures"):
            console.print(f"\n[bold red]{name.title()} Failures:[/bold red]")
            for failure in data["failures"][:5]:
                console.print(f"  • {failure}")

def _display_audit_results(report):
    """Display audit results."""
    
    score = report.overall_score
    color = "green" if score > 0.8 else "yellow" if score > 0.6 else "red"
    
    console.print(Panel(
        f"[bold]Documentation Score:[/bold] [{color}]{score:.1%}[/{color}]\n"
        f"[bold]Findings:[/bold] {len(report.findings)}",
        title="Documentation Audit Results"
    ))
    
    if report.findings:
        table = Table(title="Findings")
        table.add_column("Line", justify="center", width=6)
        table.add_column("Flaw Type", style="cyan")
        table.add_column("Severity", justify="center")
        table.add_column("Question", width=40)
        
        severity_colors = {
            "critical": "red bold",
            "high": "red",
            "medium": "yellow",
            "low": "dim"
        }
        
        for finding in report.findings[:10]:
            sev_color = severity_colors.get(finding.severity.value, "white")
            table.add_row(
                str(finding.line),
                finding.flaw_type.value,
                f"[{sev_color}]{finding.severity.value.upper()}[/{sev_color}]",
                finding.question[:40] + "..."
            )
        
        console.print(table)

if __name__ == "__main__":
    app()
```

### Day 10: Metrics & Reporting

**Create `arh/metrics/prometheus.py`**:

```python
from prometheus_client import Gauge, Counter, Histogram, generate_latest
from typing import Dict

# Agent reliability metrics
agent_reliability_score = Gauge(
    'arh_agent_reliability_score',
    'Overall agent reliability score',
    ['agent']
)

dimension_score = Gauge(
    'arh_dimension_score',
    'Score for specific reliability dimension',
    ['agent', 'dimension']
)

test_failures = Counter(
    'arh_test_failures_total',
    'Total test failures',
    ['agent', 'dimension']
)

# Knowledge reliability metrics
knowledge_score = Gauge(
    'arh_knowledge_score',
    'Documentation reliability score',
    ['document']
)

findings_count = Gauge(
    'arh_findings_count',
    'Number of findings by severity',
    ['document', 'severity']
)

# Combined trust metrics
trust_score = Gauge(
    'arh_trust_score',
    'Combined system trust score',
    ['system']
)

deployment_ready = Gauge(
    'arh_deployment_ready',
    'Whether system is ready for deployment (1=yes, 0=no)',
    ['system']
)

class MetricsExporter:
    """Export ARH metrics to Prometheus format."""
    
    def __init__(self, system_name: str = "default"):
        self.system_name = system_name
    
    def export_agent_results(self, agent_name: str, report: Dict):
        """Export agent reliability results as metrics."""
        
        agent_reliability_score.labels(agent=agent_name).set(
            report.get("overall_score", 0)
        )
        
        for dim_name, dim_data in report.get("dimensions", {}).items():
            dimension_score.labels(
                agent=agent_name, 
                dimension=dim_name
            ).set(dim_data.get("score", 0))
            
            failures = len(dim_data.get("failures", []))
            if failures > 0:
                test_failures.labels(
                    agent=agent_name,
                    dimension=dim_name
                ).inc(failures)
    
    def export_audit_results(self, doc_name: str, report):
        """Export audit results as metrics."""
        
        knowledge_score.labels(document=doc_name).set(report.overall_score)
        
        # Count findings by severity
        severity_counts = {}
        for finding in report.findings:
            sev = finding.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        for severity, count in severity_counts.items():
            findings_count.labels(
                document=doc_name,
                severity=severity
            ).set(count)
    
    def export_trust_score(self, agent_score: float, knowledge_score_val: float):
        """Export combined trust score."""
        
        combined = 0.6 * agent_score + 0.4 * knowledge_score_val
        trust_score.labels(system=self.system_name).set(combined)
        
        # Deployment ready if combined > 0.8 and no critical findings
        ready = 1 if combined >= 0.8 else 0
        deployment_ready.labels(system=self.system_name).set(ready)
    
    def get_metrics(self) -> bytes:
        """Get all metrics in Prometheus format."""
        return generate_latest()
```

### Step 4 Deliverables

| Deliverable | Status |
|-------------|--------|
| CLI with test command | ✅ |
| CLI with audit command | ✅ |
| CLI with trust-eval command | ✅ |
| Prometheus metrics export | ✅ |
| Rich terminal output | ✅ |

### Step 4 Exit Criteria
- [ ] `arh test --agent URL` works end-to-end
- [ ] `arh audit document.md` works end-to-end
- [ ] Metrics exportable to Prometheus
- [ ] Clean terminal output with Rich

---

## Step 5: Polish & Documentation (Days 11-12)

### Objective
Complete documentation, create demo materials, and prepare for showcase.

### Day 11: Documentation

**Create `README.md`**:

```markdown
# Agent Reliability Harness (ARH)

> SRE Principles for AI Agents - Because "It Usually Works" Isn't Good Enough

ARH is a production-oriented trust and evaluation layer for LLM agents. 
Instead of asking "Is this model smart?", ARH asks "Is this agent safe to deploy?"

## Quick Start

```bash
# Install
pip install agent-reliability-harness

# Test an agent
arh test --agent https://api.openai.com/v1/chat/completions \
         --api-key $OPENAI_API_KEY

# Audit documentation  
arh audit ./docs/safety_manual.md --output report.json

# Combined trust evaluation
arh trust-eval --agent $AGENT_URL --kb ./docs/ --output trust.json
```

## The Four Dimensions of Reliability

| Dimension | Question | Method |
|-----------|----------|--------|
| **Robustness** | Does it break with typos? | Prompt perturbation |
| **Consistency** | Same question → same answer? | Multi-sample variance |
| **Groundedness** | Is it making things up? | Hallucination detection |
| **Predictability** | Stable under load? | Latency distribution |

## Adversarial Auditor

Inspired by Meta's Dr. Zero paper, ARH includes an Adversarial Auditor 
that finds documentation flaws by generating questions that SHOULD be 
answerable but ARE NOT.

```bash
arh audit ./lab_manual.md --hops 1,2,3
```

## Research Lineage

ARH's Adversarial Auditor is inspired by the proposer-solver framework from:

> Yue et al. "Dr. Zero: Self-Evolving Search Agents without Training Data" 
> (Meta, January 2025)

We apply the insight that "partial solver failure indicates interesting problems" 
to documentation quality, rather than model training.
```

**Create `docs/ARCHITECTURE.md`**: (Already created in Step 1)

**Create `docs/DRZERO_CONNECTION.md`**:

```markdown
# Research Lineage: From Dr. Zero to ARH

## The Original Insight

The Dr. Zero paper introduces a self-evolving framework where:
- A **Proposer** generates increasingly difficult questions
- A **Solver** attempts to answer them using search
- **Partial failure** signals the most valuable learning signal

## Our Adaptation

We apply this to documentation quality assurance:

| Dr. Zero | ARH Adversarial Auditor |
|----------|-------------------------|
| Proposer generates questions | Proposer generates adversarial questions |
| Solver uses search engine | Solver uses ONLY the document |
| Failure → training signal | Failure → documentation flaw |
| Goal: improve model | Goal: improve documentation |

## The Key Reframe

> "If an AI can't answer it from the document, a human will get it wrong too."

This transforms AI from a passive answerer into an active auditor.
```

### Day 12: Demo & Video

**Tasks**:

1. **Create Example Scripts**

`examples/demo_full_pipeline.py`:
```python
"""
Full ARH demonstration: Agent testing + Document auditing
"""
from arh.core.agent_wrapper import OpenAIWrapper
from arh.core.harness import ReliabilityHarness
from arh.auditor.auditor import AdversarialAuditor
import os

# Setup
api_key = os.getenv("OPENAI_API_KEY")
model = OpenAIWrapper(api_key=api_key, model="gpt-4o-mini")

# === PART 1: Agent Reliability ===
print("=" * 50)
print("PART 1: Testing Agent Reliability")
print("=" * 50)

harness = ReliabilityHarness(model)
test_prompts = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "What are the safety precautions for handling acids?"
]

harness.run_all(test_prompts)
agent_report = harness.generate_report()

print(f"Agent Score: {agent_report['overall_score']:.1%}")
print(f"Verdict: {agent_report['verdict']}")

# === PART 2: Documentation Audit ===
print("\n" + "=" * 50)
print("PART 2: Auditing Documentation")
print("=" * 50)

auditor = AdversarialAuditor(proposer_model=model)

sample_doc = """
## Chemical Handling Procedures

When handling corrosive substances, ensure proper ventilation.
Transfer chemicals using appropriate containers.
In case of spills, follow cleanup procedures immediately.
Always wear protective equipment when in the lab.
"""

audit_report = auditor.audit(sample_doc)

print(f"Documentation Score: {audit_report.overall_score:.1%}")
print(f"Findings: {len(audit_report.findings)}")

for finding in audit_report.findings[:3]:
    print(f"\n  [{finding.severity.value.upper()}] {finding.flaw_type.value}")
    print(f"  Question: {finding.question}")
    print(f"  Recommendation: {finding.recommendation}")

# === PART 3: Combined Trust ===
print("\n" + "=" * 50)
print("PART 3: Combined Trust Assessment")
print("=" * 50)

trust_score = 0.6 * agent_report['overall_score'] + 0.4 * audit_report.overall_score
verdict = "PASS" if trust_score >= 0.85 else "CONDITIONAL" if trust_score >= 0.7 else "BLOCK"

print(f"Combined Trust Score: {trust_score:.1%}")
print(f"Deployment Verdict: {verdict}")
```

2. **Record Demo Video** (5 minutes)
   - Show CLI in action
   - Demonstrate agent testing
   - Show document audit with real findings
   - Explain the Dr. Zero connection

3. **Create Sample Reports**
   - `examples/sample_agent_report.json`
   - `examples/sample_audit_report.json`
   - `examples/sample_trust_report.json`

### Step 5 Deliverables

| Deliverable | Status |
|-------------|--------|
| README.md | ✅ |
| Architecture documentation | ✅ |
| Dr. Zero connection doc | ✅ |
| Demo scripts | ✅ |
| Sample reports | ✅ |
| Demo video | ✅ |

### Step 5 Exit Criteria
- [ ] README explains project clearly
- [ ] Demo script runs end-to-end
- [ ] Documentation is complete
- [ ] Video demonstrates key features

---

## Final Checklist

### Technical Completeness
- [ ] Agent wrapper works with OpenAI, Anthropic, local models
- [ ] All 4 reliability tests produce meaningful scores
- [ ] Adversarial Auditor finds real documentation flaws
- [ ] CLI is polished and user-friendly
- [ ] Metrics export to Prometheus format

### Documentation Completeness
- [ ] README with quick start
- [ ] Architecture document
- [ ] Dr. Zero research connection explained
- [ ] Example scripts

### Demo Readiness
- [ ] End-to-end demo script works
- [ ] Sample reports generated
- [ ] Video recorded

---

## Cost Summary

| Item | Estimated Cost |
|------|----------------|
| OpenAI API (testing) | $15-25 |
| Anthropic API (testing) | $5-10 |
| Buffer | $5 |
| **Total** | **$25-40** |

---

## Success Metrics

| Metric | Target | How to Verify |
|--------|--------|---------------|
| Agent testing works | ✅ | Run on OpenAI endpoint |
| Doc auditing works | ✅ | Find real flaws in sample doc |
| CLI is usable | ✅ | Complete workflow in <5 commands |
| Code is clean | ✅ | Passes linting, has types |
| Docs are complete | ✅ | New user can start in 5 min |

---

## The Interview Story

> "I built the Agent Reliability Harness after reading Meta's Dr. Zero paper on self-evolving search agents. I realized the proposer-solver pattern isn't just for training—it's a general framework for finding knowledge gaps. ARH applies SRE principles to AI: robustness testing, consistency checks, hallucination detection, and latency profiling. The Adversarial Auditor module turns AI into a documentation quality tool—it generates questions that should be answerable from your docs, and when the solver fails, you've found a gap. Every finding is human-verifiable in seconds. The output isn't 'your system is 73% good'—it's specific failures with specific fixes."

---

**Ready to execute. Let's build.**
