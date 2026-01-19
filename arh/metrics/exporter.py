"""
ARH Metrics Exporter

Export ARH metrics to Prometheus format for monitoring and alerting.
"""

from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


# Try to import prometheus_client, but make it optional
try:
    from prometheus_client import Gauge, Counter, Histogram, generate_latest, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass
class MetricSnapshot:
    """Snapshot of metrics for systems without prometheus_client."""
    timestamp: str
    agent_scores: Dict[str, float] = field(default_factory=dict)
    dimension_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    doc_scores: Dict[str, float] = field(default_factory=dict)
    findings_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    trust_score: float = 0.0
    deployment_ready: bool = False


class MetricsExporter:
    """
    Export ARH metrics to Prometheus format.
    
    Falls back to simple dict-based metrics if prometheus_client is not installed.
    """
    
    def __init__(self, system_name: str = "default"):
        """
        Initialize the metrics exporter.
        
        Args:
            system_name: Name of the system being evaluated
        """
        self.system_name = system_name
        self._snapshot = MetricSnapshot(timestamp=datetime.now().isoformat())
        
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics (only if prometheus_client is available)."""
        # Agent reliability metrics
        self.agent_reliability_score = Gauge(
            'arh_agent_reliability_score',
            'Overall agent reliability score',
            ['agent']
        )
        
        self.dimension_score = Gauge(
            'arh_dimension_score',
            'Score for specific reliability dimension',
            ['agent', 'dimension']
        )
        
        self.test_failures = Counter(
            'arh_test_failures_total',
            'Total test failures',
            ['agent', 'dimension']
        )
        
        # Knowledge reliability metrics
        self.knowledge_score = Gauge(
            'arh_knowledge_score',
            'Documentation reliability score',
            ['document']
        )
        
        self.findings_count = Gauge(
            'arh_findings_count',
            'Number of findings by severity',
            ['document', 'severity']
        )
        
        # Combined trust metrics
        self.trust_score_gauge = Gauge(
            'arh_trust_score',
            'Combined system trust score',
            ['system']
        )
        
        self.deployment_ready = Gauge(
            'arh_deployment_ready',
            'Whether system is ready for deployment (1=yes, 0=no)',
            ['system']
        )
    
    def export_agent_results(self, agent_name: str, report: Dict):
        """
        Export agent reliability results as metrics.
        
        Args:
            agent_name: Name/identifier of the agent
            report: Report dictionary from ReliabilityHarness
        """
        overall_score = report.get("overall_score", 0)
        
        # Update snapshot
        self._snapshot.agent_scores[agent_name] = overall_score
        self._snapshot.dimension_scores[agent_name] = {}
        
        if PROMETHEUS_AVAILABLE:
            self.agent_reliability_score.labels(agent=agent_name).set(overall_score)
        
        for dim_name, dim_data in report.get("dimensions", {}).items():
            dim_score = dim_data.get("score", 0)
            self._snapshot.dimension_scores[agent_name][dim_name] = dim_score
            
            if PROMETHEUS_AVAILABLE:
                self.dimension_score.labels(
                    agent=agent_name, 
                    dimension=dim_name
                ).set(dim_score)
                
                failures = len(dim_data.get("failures", []))
                if failures > 0:
                    self.test_failures.labels(
                        agent=agent_name,
                        dimension=dim_name
                    ).inc(failures)
    
    def export_audit_results(self, doc_name: str, report):
        """
        Export audit results as metrics.
        
        Args:
            doc_name: Name of the document
            report: AuditReport from AdversarialAuditor
        """
        self._snapshot.doc_scores[doc_name] = report.overall_score
        self._snapshot.findings_counts[doc_name] = {}
        
        if PROMETHEUS_AVAILABLE:
            self.knowledge_score.labels(document=doc_name).set(report.overall_score)
        
        # Count findings by severity
        severity_counts = {}
        for finding in report.findings:
            sev = finding.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        self._snapshot.findings_counts[doc_name] = severity_counts
        
        if PROMETHEUS_AVAILABLE:
            for severity, count in severity_counts.items():
                self.findings_count.labels(
                    document=doc_name,
                    severity=severity
                ).set(count)
    
    def export_trust_score(self, agent_score: float, knowledge_score_val: float):
        """
        Export combined trust score.
        
        Args:
            agent_score: Overall agent reliability score (0-1)
            knowledge_score_val: Overall knowledge/doc score (0-1)
        """
        # Weighted combination: 60% agent, 40% knowledge
        combined = 0.6 * agent_score + 0.4 * knowledge_score_val
        self._snapshot.trust_score = combined
        
        # Deployment ready if combined >= 0.8
        ready = combined >= 0.8
        self._snapshot.deployment_ready = ready
        
        if PROMETHEUS_AVAILABLE:
            self.trust_score_gauge.labels(system=self.system_name).set(combined)
            self.deployment_ready.labels(system=self.system_name).set(1 if ready else 0)
    
    def get_metrics(self) -> bytes:
        """
        Get all metrics in Prometheus format.
        
        Returns:
            Bytes containing Prometheus-formatted metrics
        """
        if PROMETHEUS_AVAILABLE:
            return generate_latest()
        else:
            # Return a simple text format without prometheus_client
            lines = [
                "# ARH Metrics (prometheus_client not installed)",
                f"# Timestamp: {self._snapshot.timestamp}",
                "",
            ]
            
            for agent, score in self._snapshot.agent_scores.items():
                lines.append(f'arh_agent_reliability_score{{agent="{agent}"}} {score}')
            
            for doc, score in self._snapshot.doc_scores.items():
                lines.append(f'arh_knowledge_score{{document="{doc}"}} {score}')
            
            lines.append(f'arh_trust_score{{system="{self.system_name}"}} {self._snapshot.trust_score}')
            lines.append(f'arh_deployment_ready{{system="{self.system_name}"}} {1 if self._snapshot.deployment_ready else 0}')
            
            return "\n".join(lines).encode('utf-8')
    
    def get_snapshot(self) -> MetricSnapshot:
        """
        Get current metrics as a snapshot object.
        
        Returns:
            MetricSnapshot with current values
        """
        self._snapshot.timestamp = datetime.now().isoformat()
        return self._snapshot
    
    def get_snapshot_dict(self) -> Dict:
        """
        Get current metrics as a dictionary.
        
        Returns:
            Dictionary with current metric values
        """
        snapshot = self.get_snapshot()
        return {
            "timestamp": snapshot.timestamp,
            "system": self.system_name,
            "agent_scores": snapshot.agent_scores,
            "dimension_scores": snapshot.dimension_scores,
            "doc_scores": snapshot.doc_scores,
            "findings_counts": snapshot.findings_counts,
            "trust_score": snapshot.trust_score,
            "deployment_ready": snapshot.deployment_ready
        }
