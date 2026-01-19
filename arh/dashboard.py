"""
ARH Premium Dashboard Output

Beautiful, Apple/Google-level terminal output using Rich.
Creates stunning visual reports that make users go "wow".
"""

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.layout import Layout
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.live import Live
from rich.style import Style
from rich.box import ROUNDED, HEAVY, DOUBLE
from rich import box
import time
from typing import Dict, List, Optional
from dataclasses import dataclass


# Premium color scheme
COLORS = {
    "primary": "#6366f1",      # Indigo
    "success": "#22c55e",       # Green
    "warning": "#f59e0b",       # Amber
    "danger": "#ef4444",        # Red
    "info": "#3b82f6",          # Blue
    "muted": "#6b7280",         # Gray
    "accent": "#8b5cf6",        # Purple
    "highlight": "#06b6d4",     # Cyan
}


def create_header(title: str = "Agent Reliability Harness") -> Panel:
    """Create a stunning header."""
    logo = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║     █████╗ ██████╗ ██╗  ██╗                                   ║
    ║    ██╔══██╗██╔══██╗██║  ██║                                   ║
    ║    ███████║██████╔╝███████║   Agent Reliability Harness       ║
    ║    ██╔══██║██╔══██╗██╔══██║   SRE for AI Agents               ║
    ║    ██║  ██║██║  ██║██║  ██║                                   ║
    ║    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝                                   ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    
    text = Text()
    text.append("╭" + "─" * 62 + "╮\n", style=COLORS["primary"])
    text.append("│", style=COLORS["primary"])
    text.append("  █████╗ ██████╗ ██╗  ██╗", style=f"bold {COLORS['accent']}")
    text.append(" " * 35 + "│\n", style=COLORS["primary"])
    text.append("│", style=COLORS["primary"])
    text.append(" ██╔══██╗██╔══██╗██║  ██║", style=f"bold {COLORS['accent']}")
    text.append(" " * 35 + "│\n", style=COLORS["primary"])
    text.append("│", style=COLORS["primary"])
    text.append(" ███████║██████╔╝███████║", style=f"bold {COLORS['accent']}")
    text.append("  Agent Reliability Harness", style=f"bold {COLORS['highlight']}")
    text.append(" " * 8 + "│\n", style=COLORS["primary"])
    text.append("│", style=COLORS["primary"])
    text.append(" ██╔══██║██╔══██╗██╔══██║", style=f"bold {COLORS['accent']}")
    text.append("  SRE for AI Agents", style=COLORS["muted"])
    text.append(" " * 17 + "│\n", style=COLORS["primary"])
    text.append("│", style=COLORS["primary"])
    text.append(" ██║  ██║██║  ██║██║  ██║", style=f"bold {COLORS['accent']}")
    text.append(" " * 35 + "│\n", style=COLORS["primary"])
    text.append("│", style=COLORS["primary"])
    text.append(" ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝", style=f"bold {COLORS['accent']}")
    text.append(" " * 35 + "│\n", style=COLORS["primary"])
    text.append("╰" + "─" * 62 + "╯", style=COLORS["primary"])
    
    return Align.center(text)


def create_score_gauge(score: float, label: str = "Score") -> Panel:
    """Create a beautiful score gauge."""
    # Determine color based on score
    if score >= 0.85:
        color = COLORS["success"]
        status = "EXCELLENT"
        emoji = "✨"
    elif score >= 0.70:
        color = COLORS["warning"]
        status = "GOOD"
        emoji = "👍"
    elif score >= 0.50:
        color = "#f97316"  # Orange
        status = "NEEDS WORK"
        emoji = "⚠️"
    else:
        color = COLORS["danger"]
        status = "CRITICAL"
        emoji = "🔴"
    
    # Create visual gauge
    filled = int(score * 20)
    empty = 20 - filled
    gauge = "█" * filled + "░" * empty
    
    content = Text()
    content.append(f"\n{emoji} ", style="bold")
    content.append(f"{score:.1%}", style=f"bold {color}")
    content.append(f" {status}\n\n", style=f"{color}")
    content.append("  [", style=COLORS["muted"])
    content.append(gauge[:filled], style=color)
    content.append(gauge[filled:], style=COLORS["muted"])
    content.append("]\n", style=COLORS["muted"])
    
    return Panel(
        Align.center(content),
        title=f"[bold {COLORS['primary']}]{label}[/]",
        border_style=color,
        box=ROUNDED,
        padding=(0, 2)
    )


def create_dimensions_table(dimensions: Dict) -> Table:
    """Create a beautiful dimensions table."""
    table = Table(
        show_header=True,
        header_style=f"bold {COLORS['primary']}",
        border_style=COLORS["muted"],
        box=ROUNDED,
        title="[bold]Reliability Dimensions[/]",
        title_style=COLORS["primary"],
        expand=True
    )
    
    table.add_column("Dimension", style="bold white")
    table.add_column("Score", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Visual", justify="left", width=20)
    
    for name, data in dimensions.items():
        score = data.get("score", 0)
        status = data.get("status", "unknown")
        
        # Score styling
        if score >= 0.85:
            score_style = COLORS["success"]
            status_text = "✅ PASS"
        elif score >= 0.70:
            score_style = COLORS["warning"]
            status_text = "⚠️  WARN"
        else:
            score_style = COLORS["danger"]
            status_text = "❌ FAIL"
        
        # Visual bar
        filled = int(score * 15)
        bar = f"[{score_style}]{'█' * filled}[/][dim]{'░' * (15 - filled)}[/]"
        
        table.add_row(
            name.capitalize(),
            f"[bold {score_style}]{score:.1%}[/]",
            status_text,
            bar
        )
    
    return table


def create_findings_panel(findings: List, max_display: int = 5) -> Panel:
    """Create a beautiful findings panel."""
    if not findings:
        return Panel(
            Align.center(Text("✨ No issues found!", style=f"bold {COLORS['success']}")),
            title="[bold]Findings[/]",
            border_style=COLORS["success"]
        )
    
    content = []
    
    # Severity icons
    sev_icons = {
        "critical": ("🔴", COLORS["danger"], "CRITICAL"),
        "high": ("🟠", "#f97316", "HIGH"),
        "medium": ("🟡", COLORS["warning"], "MEDIUM"),
        "low": ("🟢", COLORS["success"], "LOW"),
    }
    
    for i, finding in enumerate(findings[:max_display], 1):
        sev = finding.severity.value
        icon, color, label = sev_icons.get(sev, ("⚪", COLORS["muted"], sev.upper()))
        
        text = Text()
        text.append(f"\n{icon} ", style="bold")
        text.append(f"[{finding.flaw_type.value.upper()}]", style=f"bold {color}")
        text.append(f" Line {finding.line}\n", style=COLORS["muted"])
        text.append(f"   {finding.question[:60]}...\n", style="white")
        text.append(f"   💡 ", style=COLORS["info"])
        text.append(f"{finding.recommendation[:50]}...\n", style=COLORS["muted"])
        
        content.append(text)
    
    if len(findings) > max_display:
        remaining = Text(f"\n   ... and {len(findings) - max_display} more findings\n", style=COLORS["muted"])
        content.append(remaining)
    
    return Panel(
        Group(*content),
        title=f"[bold {COLORS['warning']}]📋 Findings ({len(findings)} total)[/]",
        border_style=COLORS["warning"],
        box=ROUNDED
    )


def create_summary_cards(agent_score: float, doc_score: float, trust_score: float, verdict: str) -> Columns:
    """Create summary stat cards."""
    cards = []
    
    # Agent card
    agent_color = COLORS["success"] if agent_score >= 0.85 else COLORS["warning"] if agent_score >= 0.70 else COLORS["danger"]
    agent_card = Panel(
        Align.center(Text(f"🤖\n{agent_score:.1%}", style=f"bold {agent_color}")),
        title="[bold]Agent[/]",
        border_style=agent_color,
        width=15
    )
    cards.append(agent_card)
    
    # Doc card
    doc_color = COLORS["success"] if doc_score >= 0.85 else COLORS["warning"] if doc_score >= 0.70 else COLORS["danger"]
    doc_card = Panel(
        Align.center(Text(f"📄\n{doc_score:.1%}", style=f"bold {doc_color}")),
        title="[bold]Docs[/]",
        border_style=doc_color,
        width=15
    )
    cards.append(doc_card)
    
    # Trust card
    trust_color = COLORS["success"] if trust_score >= 0.85 else COLORS["warning"] if trust_score >= 0.70 else COLORS["danger"]
    trust_card = Panel(
        Align.center(Text(f"🎯\n{trust_score:.1%}", style=f"bold {trust_color}")),
        title="[bold]Trust[/]",
        border_style=trust_color,
        width=15
    )
    cards.append(trust_card)
    
    # Verdict card
    if verdict == "PASS":
        verdict_color = COLORS["success"]
        verdict_emoji = "✅"
    elif verdict == "CONDITIONAL_PASS":
        verdict_color = COLORS["warning"]
        verdict_emoji = "⚠️"
    else:
        verdict_color = COLORS["danger"]
        verdict_emoji = "🚫"
    
    verdict_card = Panel(
        Align.center(Text(f"{verdict_emoji}\n{verdict}", style=f"bold {verdict_color}")),
        title="[bold]Verdict[/]",
        border_style=verdict_color,
        width=18
    )
    cards.append(verdict_card)
    
    return Columns(cards, equal=False, expand=True)


def create_full_dashboard(
    agent_report: Dict,
    audit_report,
    model_name: str = "unknown"
) -> None:
    """Create and display the full premium dashboard."""
    console = Console()
    
    # Clear screen for immersive experience
    console.clear()
    
    # Header
    console.print(create_header())
    console.print()
    
    # Model info
    console.print(
        Panel(
            f"[bold {COLORS['highlight']}]🤖 Model:[/] {model_name}",
            border_style=COLORS["muted"],
            box=ROUNDED
        )
    )
    console.print()
    
    # Calculate trust score
    agent_score = agent_report.get("overall_score", 0)
    doc_score = audit_report.overall_score
    trust_score = 0.6 * agent_score + 0.4 * doc_score
    
    if trust_score >= 0.85:
        verdict = "PASS"
    elif trust_score >= 0.70:
        verdict = "CONDITIONAL_PASS"
    else:
        verdict = "BLOCK"
    
    # Summary cards
    console.print(Align.center(create_summary_cards(agent_score, doc_score, trust_score, verdict)))
    console.print()
    
    # Dimensions table
    if "dimensions" in agent_report:
        console.print(create_dimensions_table(agent_report["dimensions"]))
        console.print()
    
    # Score gauge
    console.print(Columns([
        create_score_gauge(agent_score, "Agent Reliability"),
        create_score_gauge(doc_score, "Documentation Quality"),
    ], equal=True, expand=True))
    console.print()
    
    # Findings
    if hasattr(audit_report, 'findings'):
        console.print(create_findings_panel(audit_report.findings))
    
    console.print()
    
    # Footer
    footer = Panel(
        Align.center(Text("ARH v0.1.0 | SRE for AI Agents | Made with ❤️", style=COLORS["muted"])),
        border_style=COLORS["muted"],
        box=ROUNDED
    )
    console.print(footer)


def run_with_progress(func, description: str = "Processing..."):
    """Run a function with a beautiful progress spinner."""
    console = Console()
    
    with Progress(
        SpinnerColumn(spinner_name="dots12", style=COLORS["primary"]),
        TextColumn(f"[bold {COLORS['primary']}]{description}[/]"),
        BarColumn(complete_style=COLORS["primary"], finished_style=COLORS["success"]),
        TaskProgressColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(description, total=100)
        
        # Simulate progress while function runs
        import threading
        result = [None]
        error = [None]
        
        def run():
            try:
                result[0] = func()
            except Exception as e:
                error[0] = e
        
        thread = threading.Thread(target=run)
        thread.start()
        
        while thread.is_alive():
            progress.update(task, advance=1)
            if progress.tasks[0].completed >= 95:
                progress.tasks[0].completed = 50
            time.sleep(0.1)
        
        progress.update(task, completed=100)
        
        if error[0]:
            raise error[0]
        
        return result[0]


# Quick display functions
def show_success(message: str):
    """Show a success message."""
    Console().print(Panel(
        f"[bold {COLORS['success']}]✅ {message}[/]",
        border_style=COLORS["success"]
    ))


def show_error(message: str):
    """Show an error message."""
    Console().print(Panel(
        f"[bold {COLORS['danger']}]❌ {message}[/]",
        border_style=COLORS["danger"]
    ))


def show_warning(message: str):
    """Show a warning message."""
    Console().print(Panel(
        f"[bold {COLORS['warning']}]⚠️  {message}[/]",
        border_style=COLORS["warning"]
    ))


def show_info(message: str):
    """Show an info message."""
    Console().print(Panel(
        f"[bold {COLORS['info']}]ℹ️  {message}[/]",
        border_style=COLORS["info"]
    ))
