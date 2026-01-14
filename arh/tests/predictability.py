"""
ARH Predictability Test

Tests agent predictability under load.
Measures latency distribution and tail behavior.
"""

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
        """
        Initialize predictability test.
        
        Args:
            p99_slo_ms: P99 latency SLO in milliseconds
            timeout_threshold: Acceptable timeout rate (0-1)
            samples: Number of samples to collect
        """
        self.p99_slo = p99_slo_ms
        self.timeout_threshold = timeout_threshold
        self.samples = samples
    
    def run(self, agent: AgentWrapper, prompts: List[str]) -> TestResult:
        """
        Run predictability test on agent with given prompts.
        
        Args:
            agent: The agent wrapper to test
            prompts: List of prompts to test with
            
        Returns:
            TestResult with score, status, and details
        """
        latencies = []
        timeouts = 0
        
        # Cycle through prompts to get enough samples
        if not prompts:
            return TestResult(
                name="predictability",
                score=0.0,
                status=TestStatus.ERROR,
                details={"error": "No prompts provided"},
                failures=["No prompts provided for testing"]
            )
        
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
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        variance = float(np.std(latencies))
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
        """Generate recommendations based on metrics."""
        recs = []
        if p99 > self.p99_slo:
            recs.append("Consider response streaming for long generations")
            recs.append("Investigate slow queries for optimization opportunities")
        if variance > 2000:
            recs.append("High latency variance - check for resource contention")
        if timeout_rate > 0.01:
            recs.append("Implement retry logic with exponential backoff")
        return recs
