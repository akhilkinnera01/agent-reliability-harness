#!/usr/bin/env python3
"""
ARH Premium Demo with Stunning Dashboard Output

Run this to see the full premium experience!
Auto-detects your API key and shows beautiful results.
"""

import os
import sys
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.live import Live
from rich.table import Table

# Import dashboard
from arh.dashboard import (
    create_header, create_score_gauge, create_dimensions_table,
    create_findings_panel, create_summary_cards, create_full_dashboard,
    show_success, show_error, show_info, show_warning, COLORS
)

console = Console()


# Sample document
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


class MockAgent:
    """Mock agent for demo."""
    def __init__(self):
        self.model = "mock-demo"
        self.response_log = []
    
    def query(self, prompt, **kwargs):
        from arh.core.models import AgentResponse
        import random
        time.sleep(random.uniform(0.05, 0.1))
        
        if "2 + 2" in prompt or "capital" in prompt.lower():
            content = "The answer is correct."
        else:
            content = "Response generated."
        
        response = AgentResponse(content=content, latency_ms=50, model=self.model)
        self.response_log.append(response)
        return response


def detect_model():
    """Detect available API and create agent."""
    from arh.core import UniversalWrapper
    
    keys = [
        ("GEMINI_API_KEY", "gemini/gemini-2.5-flash", "Gemini 2.5 Flash"),
        ("OPENAI_API_KEY", "gpt-4o-mini", "GPT-4o Mini"),
        ("ANTHROPIC_API_KEY", "anthropic/claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"),
        ("GROQ_API_KEY", "groq/llama-3.1-70b-versatile", "Llama 3.1 70B"),
    ]
    
    for env_var, model, name in keys:
        key = os.getenv(env_var)
        if key:
            return UniversalWrapper(model=model, api_key=key), name, False
    
    return MockAgent(), "Mock Agent (Demo Mode)", True


def run_demo():
    """Run the premium demo."""
    # Clear and show header
    console.clear()
    console.print(create_header())
    console.print()
    
    # Detection phase
    show_info("Detecting available AI models...")
    time.sleep(0.5)
    
    agent, model_name, is_mock = detect_model()
    
    if is_mock:
        show_warning(f"No API key found - using {model_name}")
        console.print(f"   [dim]Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY for real testing[/]")
    else:
        show_success(f"Using {model_name}")
    
    console.print()
    time.sleep(0.5)
    
    # Run reliability tests with progress
    console.print(Panel(
        f"[bold {COLORS['primary']}]📊 PHASE 1: Agent Reliability Testing[/]",
        border_style=COLORS["primary"]
    ))
    
    from arh.core.harness import ReliabilityHarness
    harness = ReliabilityHarness(agent)
    
    prompts = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "Explain quantum computing briefly."
    ]
    
    with Progress(
        SpinnerColumn(spinner_name="dots12", style=COLORS["primary"]),
        TextColumn(f"[bold]Running reliability tests...[/]"),
        BarColumn(complete_style=COLORS["primary"], finished_style=COLORS["success"]),
        console=console,
    ) as progress:
        task = progress.add_task("Testing", total=len(prompts) * 4)
        
        # Run tests with visual progress
        for test_name in ["robustness", "consistency", "groundedness", "predictability"]:
            harness.run_test(test_name, prompts)
            progress.update(task, advance=len(prompts))
            time.sleep(0.2)
    
    agent_report = harness.generate_report()
    show_success(f"Agent Score: {agent_report['overall_score']:.1%}")
    console.print()
    time.sleep(0.3)
    
    # Run documentation audit
    console.print(Panel(
        f"[bold {COLORS['accent']}]📋 PHASE 2: Documentation Audit[/]",
        border_style=COLORS["accent"]
    ))
    
    from arh.auditor import AdversarialAuditor
    auditor = AdversarialAuditor(proposer_model=agent)
    
    with Progress(
        SpinnerColumn(spinner_name="dots12", style=COLORS["accent"]),
        TextColumn(f"[bold]Analyzing document for flaws...[/]"),
        BarColumn(complete_style=COLORS["accent"], finished_style=COLORS["success"]),
        console=console,
    ) as progress:
        task = progress.add_task("Auditing", total=100)
        
        # Run audit
        if is_mock:
            audit_report = auditor.audit_simple(SAMPLE_DOC, document_name="chemical_procedures.md")
        else:
            audit_report = auditor.audit(SAMPLE_DOC, document_name="chemical_procedures.md")
        
        progress.update(task, completed=100)
    
    show_success(f"Documentation Score: {audit_report.overall_score:.1%} ({len(audit_report.findings)} findings)")
    console.print()
    time.sleep(0.3)
    
    # Calculate combined trust
    console.print(Panel(
        f"[bold {COLORS['highlight']}]🎯 PHASE 3: Trust Assessment[/]",
        border_style=COLORS["highlight"]
    ))
    time.sleep(0.5)
    
    # Show full dashboard
    console.print()
    create_full_dashboard(agent_report, audit_report, model_name)
    
    console.print()
    console.print(f"[bold {COLORS['success']}]✨ Demo Complete![/]")
    console.print(f"[dim]Run on your own file: python3 examples/run_on_file.py <path>[/]")


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled[/]")
    except Exception as e:
        show_error(f"Error: {e}")
        raise
