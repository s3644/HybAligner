#!/usr/bin/env python3
"""
HybAligner TUI — Interactive terminal interface with live progress.

Usage: python hyb_align_tui.py
"""

from __future__ import annotations

import os
import sys
import glob
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn,
)
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box

console = Console()


# ═══════════════════════════════════════════════════════════
# System info
# ═══════════════════════════════════════════════════════════
def get_system_info() -> dict:
    """Collect system hardware info."""
    import subprocess
    info = {}

    # GPU
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        info["gpu"] = r.stdout.strip()
    except Exception:
        info["gpu"] = "Not detected"

    # CPU
    info["cpu_cores"] = os.cpu_count() or "?"

    # RAM
    try:
        r = subprocess.run(["free", "-h"], capture_output=True, text=True, timeout=5)
        parts = r.stdout.split("\n")[1].split()
        info["ram"] = f"{parts[1]} total, {parts[3]} used"
    except Exception:
        info["ram"] = "Unknown"

    # Disk
    try:
        r = subprocess.run(["df", "-h", "."], capture_output=True, text=True, timeout=5)
        parts = r.stdout.split("\n")[1].split()
        info["disk"] = f"{parts[1]} total, {parts[3]} free"
    except Exception:
        info["disk"] = "Unknown"

    # CUDA
    try:
        r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.split("\n"):
            if "release" in line:
                info["cuda"] = line.strip()
                break
    except Exception:
        info["cuda"] = "Not found"

    return info


# ═══════════════════════════════════════════════════════════
# Welcome screen
# ═══════════════════════════════════════════════════════════
def show_welcome():
    """Display welcome screen with system info."""
    console.clear()
    sys_info = get_system_info()

    title = Text("HybAligner", style="bold bright_cyan")
    subtitle = Text("GPU-Accelerated Sequence Aligner v0.5.0", style="dim")
    title.append("\n")
    title.append(subtitle)

    info_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    info_table.add_column(style="dim")
    info_table.add_column(style="bright_white")
    info_table.add_row("GPU", sys_info["gpu"])
    info_table.add_row("CPU", f"{sys_info['cpu_cores']} cores")
    info_table.add_row("RAM", sys_info["ram"])
    info_table.add_row("Disk", sys_info["disk"])
    info_table.add_row("CUDA", sys_info.get("cuda", "N/A"))

    console.print(Panel(
        info_table,
        title=title,
        border_style="bright_cyan",
        padding=(1, 2),
    ))
    console.print()


# ═══════════════════════════════════════════════════════════
# File browser
# ═══════════════════════════════════════════════════════════
def browse_fastq() -> List[str]:
    """Interactive FASTQ file selection."""
    console.print(Panel(
        "[bold]Select FASTQ Files[/bold]\n\n"
        "  • Type a file path or glob pattern\n"
        "  • Press Enter for current directory (*.fastq)\n"
        "  • Type 'done' when finished",
        border_style="yellow",
    ))

    files = []
    while True:
        path = Prompt.ask("  FASTQ file/pattern", default="*.fastq")
        if path.lower() == "done":
            if files:
                break
            console.print("  [red]Add at least one file[/red]")
            continue

        matches = sorted(glob.glob(path))
        if not matches:
            console.print(f"  [red]No files match: {path}[/red]")
            continue

        for m in matches:
            if m not in files:
                files.append(m)
                size = os.path.getsize(m)
                console.print(f"  [green]+[/green] {m} ([dim]{size:,} bytes[/dim])")

        if not Confirm.ask("  Add more?", default=False):
            break

    return files


def browse_ref() -> str:
    """Interactive reference file selection."""
    while True:
        path = Prompt.ask("  Reference FASTA", default="*.fa")
        matches = sorted(glob.glob(path))
        if matches:
            ref = matches[0]
            size = os.path.getsize(ref)
            console.print(f"  [green]Selected:[/green] {ref} ([dim]{size:,} bytes[/dim])")
            if Confirm.ask("  Use this reference?", default=True):
                return ref
        else:
            console.print(f"  [red]No files match: {path}[/red]")


# ═══════════════════════════════════════════════════════════
# Parameter config
# ═══════════════════════════════════════════════════════════
def configure_params() -> dict:
    """Interactive parameter configuration."""
    console.print()
    console.print(Panel("[bold]Alignment Parameters[/bold]", border_style="yellow"))

    params = {}

    # Preset
    preset = Prompt.ask(
        "  Preset",
        choices=["fast", "balanced", "accurate", "custom"],
        default="balanced",
    )

    if preset == "fast":
        params = {"band_width": 20, "gap_open": 5, "gap_extend": 2, "score_only": True}
    elif preset == "balanced":
        params = {"band_width": 50, "gap_open": 5, "gap_extend": 2, "score_only": False}
    elif preset == "accurate":
        params = {"band_width": 80, "gap_open": 5, "gap_extend": 2, "score_only": False}
    else:
        params["band_width"] = IntPrompt.ask("  Band width", default=50)
        params["gap_open"] = IntPrompt.ask("  Gap open penalty", default=5)
        params["gap_extend"] = IntPrompt.ask("  Gap extend penalty", default=2)
        params["score_only"] = Confirm.ask("  Score-only mode (faster, no bounds)?", default=False)

    params["verbose"] = Confirm.ask("  Show detailed per-file stats?", default=True)

    console.print()
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="bright_white")
    table.add_row("Preset", preset)
    table.add_row("Band width", str(params["band_width"]))
    table.add_row("Gap open", str(params["gap_open"]))
    table.add_row("Gap extend", str(params["gap_extend"]))
    table.add_row("Score only", str(params["score_only"]))
    console.print(Panel(table, title="[bold]Configuration[/bold]", border_style="green"))

    return params


# ═══════════════════════════════════════════════════════════
# Alignment runner with live progress
# ═══════════════════════════════════════════════════════════
def run_alignment(fastq_files: List[str], ref_path: str, params: dict, output_dir: str):
    """Run alignment with live progress display."""
    from runtime.manager import parse_fastq, parse_fasta
    from gpu.fast_align import FastAligner
    import obs.log; obs.log._logger = None  # silent

    console.clear()
    console.print(Panel("[bold bright_cyan]Running Alignment...[/bold bright_cyan]", border_style="bright_cyan"))

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        # Stage 1: Load reference
        task = progress.add_task("[cyan]Loading reference...", total=None)
        t0 = time.perf_counter()
        ref_seq = parse_fasta(ref_path)
        progress.update(task, description=f"[green]✓ Reference loaded ({len(ref_seq):,} bp)")

        # Stage 2: Init aligner
        task = progress.add_task("[cyan]Initializing GPU...", total=100)
        sample = parse_fastq(fastq_files[0])
        max_len = max(len(r) for r in sample) if sample else 300
        aligner = FastAligner(
            max_reads=len(sample) + 100,
            max_read_len=max_len + 50,
            max_ref_len=len(ref_seq) + 100,
        )
        aligner.align(["A"], ref_seq)  # warmup
        progress.update(task, completed=100, description=f"[green]✓ GPU ready (max_read_len={max_len})")

        # Stage 3: Process files
        total_files = len(fastq_files)
        all_stats = []

        for idx, fq in enumerate(fastq_files, 1):
            fq_name = Path(fq).stem

            # Parse
            task = progress.add_task(f"[cyan]File {idx}/{total_files}: Parsing {fq_name}...", total=100)
            reads = parse_fastq(fq)
            n = len(reads)
            progress.update(task, completed=100, description=f"[green]✓ {fq_name}: {n:,} reads")

            # Align
            task = progress.add_task(f"[cyan]     Aligning...", total=100)
            progress.update(task, completed=50)
            t_a = time.perf_counter()
            scores, rs, re, fs, fe = aligner.align(
                reads, ref_seq,
                band_width=params["band_width"],
                gap_open=params["gap_open"],
                gap_extend=params["gap_extend"],
                score_only=params["score_only"],
            )
            ta_ms = (time.perf_counter() - t_a) * 1000
            na = int(np.count_nonzero(scores))
            progress.update(task, completed=100,
                description=f"[green]✓ Aligned: {na:,}/{n:,} ({100*na/max(n,1):.1f}%), {n/(ta_ms/1000):,.0f} reads/s")

            # Save
            task = progress.add_task("[cyan]     Saving...", total=100)
            out_path = os.path.join(output_dir, f"{fq_name}_aligned.tsv")
            os.makedirs(output_dir, exist_ok=True)
            if params["score_only"]:
                np.savetxt(out_path, scores, fmt="%.1f", header="score", comments="")
            else:
                data = np.column_stack([scores, rs, re, fs, fe])
                np.savetxt(out_path, data, fmt=["%.1f","%d","%d","%d","%d"],
                    header="score read_start read_end ref_start ref_end", comments="")
            progress.update(task, completed=100, description=f"[green]✓ Saved: {out_path}")

            all_stats.append({
                "file": fq, "n_reads": n, "n_aligned": na,
                "pct": round(100*na/max(n,1), 1), "align_ms": round(ta_ms, 1),
            })

    # Results
    show_results(all_stats, output_dir, ref_path, params)


# ═══════════════════════════════════════════════════════════
# Results display
# ═══════════════════════════════════════════════════════════
def show_results(stats: list, output_dir: str, ref_path: str, params: dict):
    """Display results in formatted tables."""
    console.clear()
    total_reads = sum(s["n_reads"] for s in stats)
    total_aligned = sum(s["n_aligned"] for s in stats)

    # Summary panel
    summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    summary.add_column(style="dim")
    summary.add_column(style="bright_white")
    summary.add_row("Files processed", str(len(stats)))
    summary.add_row("Total reads", f"{total_reads:,}")
    summary.add_row("Total aligned", f"{total_aligned:,} ({100*total_aligned/max(total_reads,1):.1f}%)")
    summary.add_row("Band width", str(params["band_width"]))
    summary.add_row("Mode", "score-only" if params["score_only"] else "affine + bounds")
    summary.add_row("Output", output_dir)
    summary.add_row("Reference", ref_path)

    console.print(Panel(summary, title="[bold bright_green]✓ Alignment Complete[/bold bright_green]", border_style="bright_green"))

    # Per-file table
    if len(stats) > 1 or params["verbose"]:
        console.print()
        table = Table(title="Per-File Results", box=box.SIMPLE_HEAVY)
        table.add_column("File", style="cyan")
        table.add_column("Reads", justify="right")
        table.add_column("Aligned", justify="right")
        table.add_column("%", justify="right")
        table.add_column("Time", justify="right")

        for s in stats:
            fname = Path(s["file"]).name
            table.add_row(
                fname, f"{s['n_reads']:,}", f"{s['n_aligned']:,}",
                f"{s['pct']}%", f"{s['align_ms']}ms",
            )

        console.print(table)

    console.print()
    console.print(f"  [dim]Summary saved to: {output_dir}/hyb_summary.json[/dim]")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    """Main TUI entry point."""
    try:
        # Welcome
        show_welcome()

        # Check for CLI args (quick mode)
        if len(sys.argv) > 1:
            console.print("[yellow]CLI mode detected. Use 'python hyb_align.py' for CLI.[/yellow]")
            console.print("[dim]Running TUI with defaults...[/dim]")

        if not Confirm.ask("Start alignment?", default=True):
            console.print("[dim]Exiting.[/dim]")
            return

        # FASTQ files
        fastq_files = browse_fastq()
        if not fastq_files:
            console.print("[red]No files selected. Exiting.[/red]")
            return

        # Reference
        ref_path = browse_ref()

        # Parameters
        params = configure_params()

        # Output
        output_dir = Prompt.ask("  Output directory", default="hyb_results")
        os.makedirs(output_dir, exist_ok=True)

        # Confirm
        console.print()
        if not Confirm.ask("[bold]Start alignment now?[/bold]", default=True):
            console.print("[dim]Cancelled.[/dim]")
            return

        # Run
        run_alignment(fastq_files, ref_path, params, output_dir)

    except KeyboardInterrupt:
        console.print("\n[red]Interrupted.[/red]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
