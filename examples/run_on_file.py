#!/usr/bin/env python3
"""
Run ARH on Any Document

Supports: PDF, DOCX, EPUB, Markdown, HTML, TXT, and more!

Usage:
    python3 examples/run_on_file.py /path/to/document.pdf
    python3 examples/run_on_file.py /path/to/manual.docx
    python3 examples/run_on_file.py /path/to/ebook.epub
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.align import Align
from rich.text import Text

# Import ARH components
from arh.document_loader import DocumentLoader, LoadedDocument
from arh.dashboard import (
    create_header, create_score_gauge, create_findings_panel,
    create_summary_cards, show_success, show_error, show_info, 
    show_warning, COLORS
)
from arh.core.agent_wrapper import AgentWrapper
from arh.core.models import AgentResponse
from arh.auditor.auditor import AdversarialAuditor


console = Console()


class MockAgent(AgentWrapper):
    """Mock agent for simple mode."""
    def __init__(self):
        super().__init__(endpoint="mock://", model="simple")
    def query(self, prompt: str, **kwargs) -> AgentResponse:
        return AgentResponse(content="mock", latency_ms=0, model="simple")


def get_agent():
    """Auto-detect API key and return appropriate agent."""
    from arh.core import UniversalWrapper
    
    keys = [
        ("GEMINI_API_KEY", "gemini/gemini-2.5-flash", "Gemini 2.5 Flash"),
        ("OPENAI_API_KEY", "gpt-4o-mini", "GPT-4o Mini"),
        ("ANTHROPIC_API_KEY", "anthropic/claude-3-5-sonnet-20241022", "Claude 3.5"),
        ("GROQ_API_KEY", "groq/llama-3.1-70b-versatile", "Llama 3.1 70B"),
    ]
    
    for env_var, model, name in keys:
        key = os.getenv(env_var)
        if key:
            return UniversalWrapper(model=model, api_key=key), name, False
    
    return MockAgent(), "Simple Mode", True


def create_document_info_panel(doc: LoadedDocument) -> Panel:
    """Create a beautiful document info panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style=f"bold {COLORS['muted']}")
    table.add_column("Value", style="white")
    
    table.add_row("📄 File", doc.filename)
    table.add_row("📁 Format", doc.format)
    table.add_row("📖 Pages/Chapters", str(doc.pages))
    table.add_row("📝 Words", f"{doc.word_count:,}")
    table.add_row("🔤 Characters", f"{doc.char_count:,}")
    
    return Panel(
        table,
        title=f"[bold {COLORS['primary']}]Document Info[/]",
        border_style=COLORS["primary"],
        padding=(1, 2)
    )


def run_audit(file_path: str):
    """Run full audit on any document with premium dashboard."""
    
    # Clear screen and show header
    console.clear()
    console.print(create_header())
    console.print()
    
    path = Path(file_path)
    
    if not path.exists():
        show_error(f"File not found: {file_path}")
        console.print(f"\n[dim]Make sure the path is correct and the file exists.[/]")
        sys.exit(1)
    
    # Load document with progress
    show_info(f"Loading document: {path.name}")
    
    try:
        with Progress(
            SpinnerColumn(spinner_name="dots12", style=COLORS["primary"]),
            TextColumn("[bold]Extracting text from document...[/]"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task("Loading", total=None)
            doc = DocumentLoader.load(file_path)
    except Exception as e:
        show_error(f"Failed to load document: {e}")
        sys.exit(1)
    
    show_success(f"Loaded {doc.format} document")
    console.print()
    
    # Show document info
    console.print(create_document_info_panel(doc))
    console.print()
    
    # Get agent
    show_info("Detecting AI model...")
    agent, model_name, is_simple = get_agent()
    
    if is_simple:
        show_warning(f"No API key found - using {model_name}")
        console.print(f"   [dim]Set GEMINI_API_KEY for better analysis[/]")
    else:
        show_success(f"Using {model_name}")
    
    console.print()
    time.sleep(0.3)
    
    # Run audit
    console.print(Panel(
        f"[bold {COLORS['accent']}]🔍 Running Adversarial Audit[/]",
        border_style=COLORS["accent"]
    ))
    
    auditor = AdversarialAuditor(proposer_model=agent)
    
    with Progress(
        SpinnerColumn(spinner_name="dots12", style=COLORS["accent"]),
        TextColumn("[bold]Analyzing document for flaws...[/]"),
        BarColumn(complete_style=COLORS["accent"], finished_style=COLORS["success"]),
        console=console,
    ) as progress:
        task = progress.add_task("Auditing", total=100)
        
        if is_simple:
            report = auditor.audit_simple(doc.content, document_name=doc.filename)
        else:
            report = auditor.audit(doc.content, document_name=doc.filename)
        
        progress.update(task, completed=100)
    
    console.print()
    time.sleep(0.2)
    
    # Display results
    console.print(Panel(
        f"[bold {COLORS['highlight']}]📊 AUDIT RESULTS[/]",
        border_style=COLORS["highlight"]
    ))
    console.print()
    
    # Score display
    score = report.overall_score
    if score >= 0.85:
        verdict = "EXCELLENT"
        verdict_color = COLORS["success"]
        verdict_emoji = "✨"
    elif score >= 0.70:
        verdict = "GOOD"
        verdict_color = COLORS["warning"]  
        verdict_emoji = "👍"
    elif score >= 0.50:
        verdict = "NEEDS WORK"
        verdict_color = "#f97316"
        verdict_emoji = "⚠️"
    else:
        verdict = "CRITICAL"
        verdict_color = COLORS["danger"]
        verdict_emoji = "🚨"
    
    # Summary cards
    score_card = Panel(
        Align.center(Text(f"📊\n{score:.1%}", style=f"bold {verdict_color}")),
        title="[bold]Score[/]",
        border_style=verdict_color,
        width=15
    )
    
    findings_card = Panel(
        Align.center(Text(f"🔍\n{len(report.findings)}", style=f"bold {COLORS['warning']}")),
        title="[bold]Findings[/]",
        border_style=COLORS["warning"],
        width=15
    )
    
    verdict_card = Panel(
        Align.center(Text(f"{verdict_emoji}\n{verdict}", style=f"bold {verdict_color}")),
        title="[bold]Status[/]",
        border_style=verdict_color,
        width=18
    )
    
    console.print(Align.center(Columns([score_card, findings_card, verdict_card], equal=False)))
    console.print()
    
    # Score gauge
    console.print(create_score_gauge(score, "Documentation Quality"))
    console.print()
    
    # Findings breakdown
    if report.findings:
        # Severity summary table
        severity_counts = {}
        for f in report.findings:
            sev = f.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        sev_table = Table(
            title="[bold]Findings by Severity[/]",
            show_header=True,
            header_style=f"bold {COLORS['primary']}",
            border_style=COLORS["muted"]
        )
        sev_table.add_column("Severity", style="bold")
        sev_table.add_column("Count", justify="center")
        sev_table.add_column("Visual", width=20)
        
        sev_order = ["critical", "high", "medium", "low"]
        sev_icons = {
            "critical": ("🔴", COLORS["danger"]),
            "high": ("🟠", "#f97316"),
            "medium": ("🟡", COLORS["warning"]),
            "low": ("🟢", COLORS["success"]),
        }
        
        max_count = max(severity_counts.values()) if severity_counts else 1
        
        for sev in sev_order:
            if sev in severity_counts:
                count = severity_counts[sev]
                icon, color = sev_icons[sev]
                bar_len = int((count / max_count) * 15)
                bar = f"[{color}]{'█' * bar_len}[/][dim]{'░' * (15 - bar_len)}[/]"
                sev_table.add_row(f"{icon} {sev.upper()}", str(count), bar)
        
        console.print(sev_table)
        console.print()
    
    # Findings panel  
    console.print(create_findings_panel(report.findings, max_display=7))
    console.print()
    
    # Save report
    output_path = path.with_suffix('.audit_report.json')
    report_dict = auditor.generate_report_dict(report)
    report_dict["document_info"] = {
        "filename": doc.filename,
        "format": doc.format,
        "pages": doc.pages,
        "word_count": doc.word_count,
        "char_count": doc.char_count
    }
    output_path.write_text(json.dumps(report_dict, indent=2))
    
    # Footer
    console.print(Panel(
        f"[bold {COLORS['success']}]💾 Report saved to:[/] {output_path}",
        border_style=COLORS["success"]
    ))
    
    console.print()
    console.print(Panel(
        Align.center(Text("ARH v0.1.0 | SRE for AI Agents | Made with ❤️", style=COLORS["muted"])),
        border_style=COLORS["muted"]
    ))


def main():
    if len(sys.argv) < 2:
        console.clear()
        console.print(create_header())
        console.print()
        
        console.print(Panel(
            "[bold]Usage:[/]\n"
            "  python3 examples/run_on_file.py <path/to/document>\n\n"
            "[bold]Supported Formats:[/]\n"
            "  📄 PDF (.pdf)\n"
            "  📝 Word (.docx)\n"
            "  📚 EPUB (.epub)\n"
            "  📋 Markdown (.md)\n"
            "  📃 Plain Text (.txt)\n"
            "  🌐 HTML (.html)\n"
            "  💻 Code files (.py, .js, etc.)\n\n"
            "[bold]Examples:[/]\n"
            "  python3 examples/run_on_file.py manual.pdf\n"
            "  python3 examples/run_on_file.py docs/safety.docx\n"
            "  python3 examples/run_on_file.py README.md\n\n"
            "[bold]Set API Key for Better Analysis:[/]\n"
            "  export GEMINI_API_KEY='your-key'",
            title=f"[bold {COLORS['primary']}]ARH Document Auditor[/]",
            border_style=COLORS["primary"]
        ))
        sys.exit(0)
    
    file_path = sys.argv[1]
    
    try:
        run_audit(file_path)
    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled[/]")
    except Exception as e:
        show_error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
