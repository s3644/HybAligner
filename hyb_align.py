#!/usr/bin/env python3
"""
HybAligner CLI — GPU-accelerated sequence aligner for DGX Spark.

Usage:
    hyb-align reads.fastq ref.fasta -o results.json
    hyb-align *.fastq ref.fasta -o results/ --fast --band-width 20
    hyb-align dir/*.fq ref.fa -o out.json --cigar --parallel --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add project root (works both from source and installed package)
_SCRIPT_DIR = Path(__file__).resolve().parent
if (_SCRIPT_DIR / "gpu").exists():
    sys.path.insert(0, str(_SCRIPT_DIR))

from obs.log import init_logger, log
from gpu.fast_align import FastAligner
from runtime.manager import parse_fastq, parse_fasta


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------
def print_system_info():
    """Print CPU, GPU, RAM, disk summary."""
    import subprocess

    print("\n" + "═" * 60)
    print("  SYSTEM INFO")
    print("═" * 60)

    # GPU
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        gpu = r.stdout.strip()
        print(f"  GPU:      {gpu}")
    except Exception:
        print("  GPU:      Not detected")

    # CPU
    try:
        r = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.split("\n"):
            if "Model name" in line:
                cpu = line.split(":")[-1].strip()
                print(f"  CPU:      {cpu}")
                break
        # Core count
        for line in r.stdout.split("\n"):
            if line.startswith("CPU(s):"):
                cores = line.split(":")[-1].strip()
                print(f"  Cores:    {cores}")
                break
    except Exception:
        print(f"  CPU:      {os.cpu_count() or '?'} cores")

    # RAM
    try:
        r = subprocess.run(
            ["free", "-h"], capture_output=True, text=True, timeout=5,
        )
        mem_line = r.stdout.split("\n")[1]
        mem_parts = mem_line.split()
        print(f"  RAM:      {mem_parts[1]} total, {mem_parts[3]} used")
    except Exception:
        pass

    # Disk
    try:
        r = subprocess.run(
            ["df", "-h", "."], capture_output=True, text=True, timeout=5,
        )
        disk_line = r.stdout.split("\n")[1]
        disk_parts = disk_line.split()
        print(f"  Disk:     {disk_parts[1]} total, {disk_parts[3]} free")
    except Exception:
        pass

    # CUDA
    try:
        r = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=5,
        )
        for line in r.stdout.split("\n"):
            if "release" in line:
                print(f"  CUDA:     {line.strip()}")
                break
    except Exception:
        print("  CUDA:     Not found")

    print("═" * 60 + "\n")


# ---------------------------------------------------------------------------
# Progress stages
# ---------------------------------------------------------------------------
def progress_stage(desc: str, width: int = 60):
    """Print a stage header."""
    print(f"  {'─' * 3} {desc:<{width-6}}", end="", flush=True)


def progress_done(elapsed_ms: float = 0, extra: str = ""):
    """Complete a stage."""
    if elapsed_ms > 0:
        print(f" ✓  ({elapsed_ms:.0f}ms{extra})")
    else:
        print(" ✓")


# ---------------------------------------------------------------------------
# Alignment runner
# ---------------------------------------------------------------------------
def align_file(
    fastq_path: str,
    ref_seq: str,
    aligner: FastAligner,
    band_width: int,
    gap_open: int,
    gap_extend: int,
    score_only: bool,
    out_path: str,
) -> dict:
    """Align one FASTQ file and return stats."""

    # Stage 1: Parse
    t0 = time.perf_counter()
    reads = parse_fastq(fastq_path)
    t_parse = (time.perf_counter() - t0) * 1000

    if not reads:
        return {"file": fastq_path, "n_reads": 0, "error": "No reads found"}

    # Stage 2: Align
    t0 = time.perf_counter()
    scores, rs, re, fs, fe = aligner.align(
        reads, ref_seq,
        band_width=band_width,
        gap_open=gap_open,
        gap_extend=gap_extend,
        score_only=score_only,
    )
    t_align = (time.perf_counter() - t0) * 1000

    n_aligned = int(np.count_nonzero(scores))

    # Stage 3: Save
    t0 = time.perf_counter()
    if score_only:
        np.savetxt(out_path, scores, fmt="%.1f", header="score", comments="")
    else:
        data = np.column_stack([scores, rs, re, fs, fe])
        np.savetxt(
            out_path, data, fmt=["%.1f", "%d", "%d", "%d", "%d"],
            header="score read_start read_end ref_start ref_end", comments="",
        )
    t_save = (time.perf_counter() - t0) * 1000

    total = t_parse + t_align + t_save

    return {
        "file": fastq_path,
        "output": out_path,
        "n_reads": len(reads),
        "n_aligned": n_aligned,
        "pct_aligned": round(100.0 * n_aligned / len(reads), 1) if reads else 0,
        "score_mean": round(float(np.mean(scores)), 2) if len(scores) else 0,
        "score_max": round(float(np.max(scores)), 2) if len(scores) else 0,
        "time_parse_ms": round(t_parse, 1),
        "time_align_ms": round(t_align, 1),
        "time_save_ms": round(t_save, 1),
        "time_total_ms": round(total, 1),
        "throughput_reads_per_sec": round(len(reads) / (total / 1000), 1) if total > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="HybAligner — GPU-Accelerated Sequence Aligner for DGX Spark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hyb-align reads.fastq ref.fasta -o results.json
  hyb-align sample1.fq sample2.fq ref.fa -o out/ --fast --band-width 20
  hyb-align *.fastq ref.fasta -o out/ --cigar --parallel --verbose
  hyb-align data/*.fq ref.fa -o scores/ --score-only --band-width 50
        """,
    )
    parser.add_argument(
        "fastq", nargs="+", help="One or more FASTQ files",
    )
    parser.add_argument(
        "ref", help="Reference FASTA file",
    )

    # Alignment parameters
    g = parser.add_argument_group("Alignment")
    g.add_argument("-w", "--band-width", type=int, default=50,
                   help="Half-band width for SW DP (default: 50)")
    g.add_argument("--gap-open", type=int, default=5,
                   help="Gap opening penalty (default: 5)")
    g.add_argument("--gap-extend", type=int, default=2,
                   help="Gap extension penalty (default: 2)")
    g.add_argument("--score-only", action="store_true",
                   help="Score-only mode (faster, no alignment bounds)")

    # Output
    g = parser.add_argument_group("Output")
    g.add_argument("-o", "--output", type=str, default="hyb_results",
                   help="Output file or directory (default: hyb_results)")
    g.add_argument("--prefix", type=str, default="",
                   help="Prefix for output filenames")
    g.add_argument("-v", "--verbose", action="store_true",
                   help="Show detailed per-file stats")
    g.add_argument("--json", action="store_true",
                   help="Output summary as JSON to stdout")
    g.add_argument("-q", "--quiet", action="store_true",
                   help="Suppress all output except errors")

    # Performance
    g = parser.add_argument_group("Performance")
    g.add_argument("--system-info", action="store_true",
                   help="Print system info before alignment")
    g.add_argument("--no-progress", action="store_true",
                   help="Disable progress output")

    args = parser.parse_args()

    # --- System info ---
    if args.system_info:
        print_system_info()

    # --- Validate inputs ---
    for fq in args.fastq:
        if not os.path.exists(fq):
            print(f"Error: FASTQ file not found: {fq}", file=sys.stderr)
            sys.exit(1)

    if not os.path.exists(args.ref):
        print(f"Error: Reference FASTA not found: {args.ref}", file=sys.stderr)
        sys.exit(1)

    # --- Load reference ---
    if not args.quiet:
        print("═" * 60)
        print("  HybAligner — GPU-Accelerated Sequence Aligner")
        print("═" * 60)
        print(f"  FASTQ files:  {len(args.fastq)}")
        print(f"  Reference:    {args.ref}")
        print(f"  Band width:   {args.band_width}")
        mode = "score-only" if args.score_only else "affine + bounds"
        print(f"  Mode:         {mode}")
        print(f"  Output:       {args.output}")
        print("═" * 60)
        print()

    # Stage 1: Load reference
    if not args.quiet:
        t0 = time.perf_counter()
        progress_stage("Loading reference...")
    ref_seq = parse_fasta(args.ref)
    if not args.quiet:
        progress_done((time.perf_counter() - t0) * 1000,
                       extra=f", {len(ref_seq):,} bp")

    # Stage 2: Init aligner
    if not args.quiet:
        t0 = time.perf_counter()
        progress_stage("Initializing GPU aligner...")

    # Determine max read length from first file
    sample_reads = parse_fastq(args.fastq[0])
    max_read_len = max(len(r) for r in sample_reads) if sample_reads else 300
    total_reads = len(sample_reads)

    aligner = FastAligner(
        max_reads=total_reads + 100,
        max_read_len=max_read_len + 50,
        max_ref_len=len(ref_seq) + 100,
    )
    aligner.align(["A"], ref_seq)  # warmup

    if not args.quiet:
        progress_done((time.perf_counter() - t0) * 1000,
                       extra=f", max_read_len={max_read_len}")

    # Stage 3: Process each FASTQ
    all_results = []
    output_dir = args.output
    is_dir = len(args.fastq) > 1 or os.path.isdir(output_dir) or output_dir.endswith("/")

    if is_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Single output file: ensure parent dir exists
        parent = os.path.dirname(output_dir)
        if parent:
            os.makedirs(parent, exist_ok=True)

    for idx, fq in enumerate(args.fastq, 1):
        fq_name = Path(fq).stem
        if is_dir:
            out_name = os.path.join(output_dir, f"{args.prefix}{fq_name}_aligned.tsv")
        else:
            out_name = output_dir

        if not args.quiet:
            print()
            print(f"  [{idx}/{len(args.fastq)}] {fq}")
            progress_stage("Parsing reads...")

        t_start = time.perf_counter()
        result = align_file(
            fq, ref_seq, aligner,
            args.band_width, args.gap_open, args.gap_extend,
            args.score_only, out_name,
        )
        elapsed = (time.perf_counter() - t_start) * 1000
        all_results.append(result)

        if not args.quiet:
            progress_done(elapsed, extra=f", {result['n_reads']:,} reads")

            if args.verbose:
                print(f"    Aligned:     {result['n_aligned']:,}/{result['n_reads']:,} "
                      f"({result['pct_aligned']}%)")
                print(f"    Mean score:  {result['score_mean']}")
                print(f"    Max score:   {result['score_max']}")
                print(f"    Throughput:  {result['throughput_reads_per_sec']:,.0f} reads/s")
                print(f"    Parse:       {result['time_parse_ms']:.0f}ms")
                print(f"    Align:       {result['time_align_ms']:.0f}ms")
                print(f"    Save:        {result['time_save_ms']:.0f}ms")
                print(f"    Output:      {out_name}")

    # Stage 4: Summary
    total_reads = sum(r["n_reads"] for r in all_results)
    total_aligned = sum(r["n_aligned"] for r in all_results)
    total_ms = sum(r["time_total_ms"] for r in all_results)

    summary = {
        "tool": "HybAligner v0.5.0",
        "n_files": len(args.fastq),
        "total_reads": total_reads,
        "total_aligned": total_aligned,
        "pct_aligned": round(100.0 * total_aligned / total_reads, 1) if total_reads else 0,
        "total_time_ms": round(total_ms, 1),
        "throughput_reads_per_sec": round(total_reads / (total_ms / 1000), 1) if total_ms > 0 else 0,
        "band_width": args.band_width,
        "mode": "score-only" if args.score_only else "affine + bounds",
        "reference": args.ref,
        "files": all_results,
    }

    if not args.quiet:
        print()
        print("═" * 60)
        print("  RESULTS")
        print("═" * 60)
        print(f"  Files:          {len(args.fastq)}")
        print(f"  Total reads:    {total_reads:,}")
        print(f"  Total aligned:  {total_aligned:,} ({summary['pct_aligned']}%)")
        print(f"  Total time:     {total_ms:.0f}ms")
        print(f"  Throughput:     {summary['throughput_reads_per_sec']:,.0f} reads/s")
        print(f"  Output:         {output_dir}")
        print("═" * 60)

    # Save summary
    summary_path = os.path.join(output_dir if is_dir else ".", "hyb_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    if not args.quiet:
        print(f"\n  Summary saved to: {summary_path}")

    if args.json:
        print(json.dumps(summary, indent=2))

    return summary


if __name__ == "__main__":
    main()
