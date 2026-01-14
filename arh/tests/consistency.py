"""
ARH Consistency Test

Tests agent consistency by querying the same prompt multiple times.
A consistent agent produces semantically similar responses.
"""

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
        """
        Initialize consistency test.
        
        Args:
            samples: Number of times to query each prompt
            threshold: Pass threshold (0-1)
            temperature: Temperature to use for queries
        """
        self.samples = samples
        self.threshold = threshold
        self.temperature = temperature
    
    def run(self, agent: AgentWrapper, prompts: List[str]) -> TestResult:
        """
        Run consistency test on agent with given prompts.
        
        Args:
            agent: The agent wrapper to test
            prompts: List of prompts to test with
            
        Returns:
            TestResult with score, status, and details
        """
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
            
            if len(contents) < 2:
                # Not enough successful responses
                continue
                
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
                "average_variance": float(avg_variance),
                "samples_per_prompt": self.samples,
                "prompts_tested": len(prompts)
            },
            failures=failures[:10],
            recommendations=self._get_recommendations(score)
        )
    
    def _calculate_variance(self, responses: List[str]) -> float:
        """
        Calculate semantic variance across responses.
        
        Uses pairwise Jaccard distance as a simple metric.
        """
        if len(responses) < 2:
            return 0.0
        
        # Simple approach: pairwise Jaccard distance
        distances = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                dist = self._jaccard_distance(responses[i], responses[j])
                distances.append(dist)
        
        return float(np.mean(distances)) if distances else 0.0
    
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
        """Generate recommendations based on score."""
        recs = []
        if score < self.threshold:
            recs.append("Consider lowering temperature for more deterministic outputs")
            recs.append("Implement response caching for identical queries")
        return recs
