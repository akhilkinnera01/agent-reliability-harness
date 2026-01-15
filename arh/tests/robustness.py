"""
ARH Robustness Test

Tests agent robustness through prompt perturbations.
A robust agent should produce semantically similar outputs
despite surface-level changes to inputs.
"""

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
        """
        Initialize robustness test.
        
        Args:
            perturbations: Types of perturbations to apply
            threshold: Pass threshold (0-1)
            samples_per_perturbation: Number of samples per perturbation type
        """
        self.perturbations = perturbations or [
            "typo", "rephrase", "case_shift", "noise", "truncate"
        ]
        self.threshold = threshold
        self.samples = samples_per_perturbation
    
    def run(self, agent: AgentWrapper, prompts: List[str]) -> TestResult:
        """
        Run robustness test on agent with given prompts.
        
        Args:
            agent: The agent wrapper to test
            prompts: List of prompts to test with
            
        Returns:
            TestResult with score, status, and details
        """
        total_tests = 0
        consistent_tests = 0
        failures = []
        
        for prompt in prompts:
            # Get baseline response
            baseline = agent.query(prompt)
            
            # Skip if baseline failed
            if baseline.error:
                continue
            
            for perturb_type in self.perturbations:
                perturbed = self._apply_perturbation(prompt, perturb_type)
                response = agent.query(perturbed)
                
                # Skip if perturbed query failed
                if response.error:
                    continue
                
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
        """
        Check if two responses are semantically consistent.
        
        Uses Jaccard similarity as a simple heuristic.
        In production, consider using embedding similarity.
        """
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
