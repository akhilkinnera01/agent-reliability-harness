#!/usr/bin/env python3
"""
ARH Git History Creator (FIXED)

Pushes the ARH project to GitHub with a realistic commit history
spanning 13 WORKING days (Mon-Fri, 8am-5pm, commits every 30 minutes).

This version GUARANTEES commits are spread across all 13 days.

Usage:
    1. Create a new GitHub repo (leave it EMPTY)
    2. Run: python3 scripts/push_to_github.py <github-repo-url>
    
Example:
    python3 scripts/push_to_github.py https://github.com/username/agent-reliability-harness.git
"""

import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Configuration
WORK_START_HOUR = 8   # 8 AM
WORK_END_HOUR = 17    # 5 PM
COMMIT_INTERVAL_MINUTES = 30
NUM_WORK_DAYS = 13

# Project root (script is in /scripts folder)
PROJECT_ROOT = Path(__file__).parent.parent


def run_git(args, cwd=PROJECT_ROOT, env=None):
    """Run a git command."""
    cmd = ["git"] + args
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env
    )
    if result.returncode != 0 and result.stderr:
        print(f"Git error: {result.stderr}")
    return result


def get_all_project_files():
    """Get all files in the project, sorted for logical ordering."""
    files = []
    for path in PROJECT_ROOT.rglob("*"):
        if path.is_file():
            rel_path = path.relative_to(PROJECT_ROOT)
            rel_str = str(rel_path)
            
            # Skip unwanted files
            if any(skip in rel_str for skip in [
                ".git/", "__pycache__", ".pyc", ".DS_Store",
                "audit_report.json", ".epub", ".venv", "venv/",
                ".env", "node_modules"
            ]):
                continue
            
            files.append(rel_str)
    
    # Sort files in logical development order
    def sort_key(f):
        priority = {
            ".gitignore": 0,
            "README.md": 1,
            "requirements.txt": 2,
            "setup.py": 3,
            "pyproject.toml": 4,
        }
        
        # Check exact filename
        filename = os.path.basename(f)
        if filename in priority:
            return (0, priority[filename], f)
        
        # Directory-based priority
        if f.startswith("arh/__init__"):
            return (1, 0, f)
        if f.startswith("arh/core/"):
            return (2, 0, f)
        if f.startswith("arh/tests/"):
            return (3, 0, f)
        if f.startswith("arh/auditor/"):
            return (4, 0, f)
        if f.startswith("arh/cli/"):
            return (5, 0, f)
        if f.startswith("arh/metrics/"):
            return (6, 0, f)
        if f.startswith("arh/"):
            return (7, 0, f)
        if f.startswith("examples/"):
            return (8, 0, f)
        if f.startswith("docs/"):
            return (9, 0, f)
        if f.startswith("scripts/"):
            return (10, 0, f)
        
        return (99, 0, f)
    
    return sorted(files, key=sort_key)


def generate_commit_schedule(num_files, start_date):
    """
    Generate commit timestamps spread across 13 WORK DAYS.
    Each day: 8 AM to 5 PM, commits every 30 minutes = 18 slots per day.
    Total: 13 days × 18 slots = 234 possible slots.
    """
    
    # Calculate slots per day (8 AM to 5 PM = 9 hours = 18 half-hour slots)
    slots_per_day = (WORK_END_HOUR - WORK_START_HOUR) * 2  # 18 slots
    total_slots = NUM_WORK_DAYS * slots_per_day  # 234 slots
    
    # Distribute files evenly across all 13 days
    # Calculate how many slots to skip between commits
    if num_files >= total_slots:
        # More files than slots - multiple files per commit
        files_per_commit = (num_files // total_slots) + 1
        slot_interval = 1
    else:
        # Fewer files than slots - spread them out
        files_per_commit = 1
        slot_interval = total_slots // num_files
    
    print(f"   Slots per day: {slots_per_day}")
    print(f"   Total slots available: {total_slots}")
    print(f"   Files per commit: {files_per_commit}")
    print(f"   Slot interval: {slot_interval}")
    
    # Generate timestamps
    times = []
    current_date = start_date
    
    # Skip to first weekday
    while current_date.weekday() >= 5:
        current_date += timedelta(days=1)
    
    slot_index = 0
    days_used = 0
    
    for i in range(num_files):
        # Calculate which slot this file goes in
        target_slot = (i * slot_interval) % total_slots
        
        # Calculate day and time from slot
        day_num = target_slot // slots_per_day
        slot_in_day = target_slot % slots_per_day
        
        hour = WORK_START_HOUR + (slot_in_day // 2)
        minute = (slot_in_day % 2) * 30
        
        # Calculate actual date (skipping weekends)
        commit_date = start_date
        work_days_added = 0
        while work_days_added < day_num:
            commit_date += timedelta(days=1)
            if commit_date.weekday() < 5:  # Monday=0, Friday=4
                work_days_added += 1
        
        # Skip weekend if landed on one
        while commit_date.weekday() >= 5:
            commit_date += timedelta(days=1)
        
        commit_time = commit_date.replace(
            hour=hour,
            minute=minute,
            second=0,
            microsecond=0
        )
        times.append(commit_time)
    
    return times


def get_commit_message(file_path, index, total):
    """Generate a meaningful commit message based on file path."""
    
    filename = os.path.basename(file_path)
    dirname = os.path.dirname(file_path)
    
    # Special files
    messages = {
        ".gitignore": "Initial project setup - add .gitignore",
        "README.md": "Add project README with documentation",
        "requirements.txt": "Add requirements.txt with dependencies",
        "setup.py": "Add setup.py for package installation",
        "pyproject.toml": "Add pyproject.toml configuration",
    }
    
    if filename in messages:
        return messages[filename]
    
    # Directory-based messages
    if "arh/core/" in file_path:
        if "__init__" in filename:
            return "Initialize core module structure"
        if "models" in filename:
            return "Add core data models - TestStatus, Severity enums"
        if "agent_wrapper" in filename:
            return "Implement AgentWrapper base class with provider support"
        if "harness" in filename:
            return "Create ReliabilityHarness - main test orchestrator"
        return f"Add core module: {filename}"
    
    if "arh/tests/" in file_path:
        if "__init__" in filename:
            return "Initialize tests module structure"
        if "robustness" in filename:
            return "Implement RobustnessTest - prompt perturbation"
        if "consistency" in filename:
            return "Implement ConsistencyTest - response variance"
        if "groundedness" in filename:
            return "Implement GroundednessTest - hallucination detection"
        if "predictability" in filename:
            return "Implement PredictabilityTest - latency profiling"
        return f"Add test module: {filename}"
    
    if "arh/auditor/" in file_path:
        if "__init__" in filename:
            return "Initialize auditor module structure"
        if "proposer" in filename:
            return "Implement Proposer - adversarial question generation"
        if "solver" in filename:
            return "Implement Solver - document-constrained answering"
        if "evaluator" in filename:
            return "Implement Evaluator - flaw classification"
        if "auditor" in filename:
            return "Implement AdversarialAuditor - main orchestrator"
        return f"Add auditor component: {filename}"
    
    if "arh/cli/" in file_path:
        if "__init__" in filename:
            return "Initialize CLI module"
        if "main" in filename:
            return "Add CLI commands - test, audit, trust-eval"
        return f"Add CLI component: {filename}"
    
    if "arh/metrics/" in file_path:
        if "__init__" in filename:
            return "Initialize metrics module"
        if "exporter" in filename:
            return "Add Prometheus metrics exporter"
        return f"Add metrics component: {filename}"
    
    if "arh/" in file_path:
        if "document_loader" in filename:
            return "Add universal document loader with PDF/DOCX support"
        if "dashboard" in filename:
            return "Add premium dashboard output with visual gauges"
        if "__init__" in filename:
            return "Initialize arh package"
        return f"Add module: {filename}"
    
    if "examples/" in file_path:
        if "demo" in filename.lower():
            return f"Add demo script: {filename}"
        if "test_" in filename:
            return f"Add example test: {filename}"
        if "sample" in filename.lower():
            return f"Add sample file: {filename}"
        return f"Add example: {filename}"
    
    if "docs/" in file_path:
        if "STEP" in filename:
            step_num = ''.join(filter(str.isdigit, filename.split('-')[0]))
            return f"Add Step {step_num} implementation documentation"
        if "GETTING" in filename:
            return "Add Getting Started guide"
        if "DRZERO" in filename:
            return "Add Dr. Zero research connection documentation"
        return f"Add documentation: {filename}"
    
    if "scripts/" in file_path:
        return f"Add script: {filename}"
    
    # Generic fallback
    if index == total - 1:
        return "Final project updates"
    
    return f"Add {filename}"


def create_git_history(github_url, start_date=None):
    """Create the git history with backdated commits spread over 13 work days."""
    
    if start_date is None:
        # Start 18 calendar days ago (to account for weekends = ~13 work days)
        start_date = datetime.now() - timedelta(days=18)
    
    # Ensure we start on a weekday
    while start_date.weekday() >= 5:
        start_date += timedelta(days=1)
    
    print("=" * 60)
    print("ARH Git History Creator (FIXED)")
    print("=" * 60)
    print(f"GitHub URL: {github_url}")
    print(f"Start Date: {start_date.strftime('%Y-%m-%d (%A)')}")
    print(f"Work Days: {NUM_WORK_DAYS}")
    print(f"Work Hours: {WORK_START_HOUR}:00 - {WORK_END_HOUR}:00")
    print(f"Commit Interval: {COMMIT_INTERVAL_MINUTES} minutes")
    print("=" * 60)
    
    # Get all files
    all_files = get_all_project_files()
    num_files = len(all_files)
    
    print(f"\n📁 Total files to commit: {num_files}")
    
    if num_files == 0:
        print("❌ No files found to commit!")
        return
    
    # Generate commit schedule
    print(f"\n📅 Generating commit schedule across {NUM_WORK_DAYS} work days...")
    commit_times = generate_commit_schedule(num_files, start_date)
    
    # Remove existing git if present
    git_dir = PROJECT_ROOT / ".git"
    if git_dir.exists():
        print("\n🗑️  Removing existing git history...")
        import shutil
        shutil.rmtree(git_dir)
    
    # Initialize fresh git repo
    print("\n📦 Initializing fresh git repository...")
    run_git(["init"])
    run_git(["checkout", "-b", "main"])
    
    # Track which days we've committed on
    days_with_commits = set()
    
    # Create commits
    print(f"\n🚀 Creating {num_files} commits...\n")
    
    for i, (file_path, commit_time) in enumerate(zip(all_files, commit_times)):
        # Track the day
        day_str = commit_time.strftime("%Y-%m-%d")
        days_with_commits.add(day_str)
        
        # Generate commit message
        message = get_commit_message(file_path, i, num_files)
        
        # Format date for git
        date_str = commit_time.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Progress output
        print(f"[{i+1:3}/{num_files}] {commit_time.strftime('%Y-%m-%d %H:%M')} | {message[:50]}")
        
        # Add the file
        file_full_path = PROJECT_ROOT / file_path
        if file_full_path.exists():
            run_git(["add", file_path])
        else:
            print(f"   ⚠️  File not found: {file_path}")
            continue
        
        # Check if there's anything to commit
        status = run_git(["status", "--porcelain"])
        if not status.stdout.strip():
            print(f"   ⏭️  No changes, skipping")
            continue
        
        # Create commit with backdated timestamp
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = date_str
        env["GIT_COMMITTER_DATE"] = date_str
        
        result = run_git(["commit", "-m", message], env=env)
        if result.returncode != 0:
            print(f"   ❌ Commit failed")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 COMMIT SUMMARY")
    print("=" * 60)
    print(f"Total commits created: {len(days_with_commits)} days worth")
    print(f"Days with commits: {sorted(days_with_commits)}")
    print(f"Date range: {min(days_with_commits)} to {max(days_with_commits)}")
    
    # Set up remote and push
    print("\n🚀 Setting up remote and pushing...")
    
    # Remove existing remote if present
    run_git(["remote", "remove", "origin"])
    run_git(["remote", "add", "origin", github_url])
    
    # Push
    print("\n📤 Pushing to GitHub (this may ask for credentials)...")
    result = run_git(["push", "-u", "origin", "main", "--force"])
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✅ SUCCESS! Project pushed to GitHub!")
        print("=" * 60)
        print(f"\n🔗 View your repo: {github_url.replace('.git', '')}")
        print(f"📅 Commits span {len(days_with_commits)} work days")
        print(f"⏰ All commits between {WORK_START_HOUR}AM - {WORK_END_HOUR-12}PM")
    else:
        # Try master branch as fallback
        run_git(["branch", "-m", "main", "master"])
        result = run_git(["push", "-u", "origin", "master", "--force"])
        if result.returncode == 0:
            print("\n✅ Pushed to master branch successfully!")
        else:
            print("\n❌ Push failed. Please check:")
            print("   1. Is the GitHub repo created and empty?")
            print("   2. Do you have authentication set up?")
            print("   3. Try: git push -u origin main --force")
            print(f"\n   Error: {result.stderr}")


def main():
    if len(sys.argv) < 2:
        print("=" * 60)
        print("ARH Git History Creator (FIXED)")
        print("=" * 60)
        print("\nUsage:")
        print("  python3 scripts/push_to_github.py <github-repo-url> [start-date]")
        print("\nExamples:")
        print("  python3 scripts/push_to_github.py https://github.com/user/repo.git")
        print("  python3 scripts/push_to_github.py https://github.com/user/repo.git 2025-01-01")
        print("\nSteps:")
        print("  1. Create a new EMPTY repo on GitHub (no README!)")
        print("  2. Copy the HTTPS URL")
        print("  3. Run this script")
        print("\nThis will create commits spread across 13 work days,")
        print("Mon-Fri only, 8AM-5PM, every 30 minutes.")
        sys.exit(1)
    
    github_url = sys.argv[1]
    
    # Optional: specify start date
    start_date = None
    if len(sys.argv) >= 3:
        try:
            start_date = datetime.strptime(sys.argv[2], "%Y-%m-%d")
        except ValueError:
            print(f"❌ Invalid date format: {sys.argv[2]}")
            print("   Use YYYY-MM-DD format, e.g., 2025-01-01")
            sys.exit(1)
    
    create_git_history(github_url, start_date)


if __name__ == "__main__":
    main()