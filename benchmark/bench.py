#!/usr/bin/env python3
"""Benchmark HybAligner vs Minimap2 CPU baseline.

Generates synthetic test data, runs both aligners, and compares:
  - Throughput (reads/sec)
  - Wall-clock time
  - Score distributions
  - Alignment count

Usage:
    python benchmark/bench.py [--reads N] [--read-len L] [--ref-len R]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from obs.log import init_logger, log


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
DNA = "ACGT"


def generate_reference(length: int, seed: int = 42) -> str:
    """Generate a random DNA reference sequence."""
    random.seed(seed)
    return ''.join(random.choice(DNA) for _ in range(length))


def mutate_read(ref: str, read_len: int, error_rate: float = 0.02) -> str:
    """Extract a read from reference and introduce substitutions/indels."""
    if len(ref) < read_len + 50:
        return ref[:read_len]

    start = random.randint(0, len(ref) - read_len - 10)
    read = list(ref[start:start + read_len])

    for i in range(len(read)):
        if random.random() < error_rate:
            r = random.random()
            if r < 0.7:
                # substitution
                read[i] = random.choice(DNA.replace(read[i], ''))
            elif r < 0.85:
                # deletion
                read[i] = ''
            else:
                # insertion
                read.insert(i, random.choice(DNA))

    return ''.join(read)[:read_len]


def generate_fastq(
    ref: str, n_reads: int, read_len: int, seed: int = 123,
) -> List[str]:
    """Generate synthetic FASTQ reads from a reference."""
    random.seed(seed)
    reads = []
    for i in range(n_reads):
        read = mutate_read(ref, read_len)
        reads.append(read)
    return reads


def write_fastq(reads: List[str], path: str):
    """Write reads to FASTQ format."""
    with open(path, 'w') as f:
        for i, read in enumerate(reads):
            f.write(f"@read_{i}\n")
            f.write(f"{read}\n")
            f.write("+\n")
            f.write("I" * len(read) + "\n")


def write_fasta(seq: str, path: str, name: str = "ref"):
    """Write sequence to FASTA format."""
    with open(path, 'w') as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i + 80] + "\n")


# ---------------------------------------------------------------------------
# Benchmark result
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    name: str
    n_reads: int
    read_len: int
    ref_len: int
    elapsed_ms: float
    throughput_reads_per_sec: float
    n_aligned: int = 0
    score_mean: float = 0.0
    score_max: float = 0.0
    extra: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Minimap2 runner
# ---------------------------------------------------------------------------
def run_minimap2(fastq_path: str, ref_path: str) -> BenchResult:
    """Run minimap2 and parse PAF output for scores."""
    t0 = time.perf_counter()

    # Run minimap2 with -c to get CIGAR, -t 16 for multi-core
    result = subprocess.run(
        [
            "minimap2", "-c", "-t", "16",
            ref_path, fastq_path,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    elapsed = (time.perf_counter() - t0) * 1000.0

    # Count reads in FASTQ
    with open(fastq_path) as f:
        n_reads = sum(1 for _ in f) // 4

    # Parse PAF lines: each line is one alignment
    scores = []
    n_aligned = 0
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        fields = line.split('\t')
        if len(fields) >= 12:
            n_aligned += 1
            # Field 11 (0-based) = number of matching bases
            try:
                matches = int(fields[9])
                read_len = int(fields[1])
                # Crude score approximation: matches * 2 - (read_len - matches) * 3
                score = matches * 2 - (read_len - matches) * 3
                scores.append(float(score))
            except (ValueError, IndexError):
                scores.append(0.0)

    return BenchResult(
        name="minimap2",
        n_reads=n_reads,
        read_len=0,  # determined from data
        ref_len=0,
        elapsed_ms=elapsed,
        throughput_reads_per_sec=round(n_reads / (elapsed / 1000.0), 1),
        n_aligned=n_aligned,
        score_mean=round(float(np.mean(scores)), 2) if scores else 0,
        score_max=round(float(np.max(scores)), 2) if scores else 0,
        extra={"tool": "minimap2", "version": "2.26"},
    )


# ---------------------------------------------------------------------------
# HybAligner runner (standard scheduler-based)
# ---------------------------------------------------------------------------
def run_hybaligner(
    fastq_path: str,
    ref_path: str,
    batch_size: int = 1024,
    band_width: int = 50,
    streams: bool = False,
    seed: bool = True,
) -> BenchResult:
    """Run HybAligner pipeline and extract summary."""
    from runtime.manager import run_pipeline

    t0 = time.perf_counter()
    summary = run_pipeline(
        fastq_path=fastq_path,
        ref_path=ref_path,
        batch_size=batch_size,
        gpu_only=True,
        band_width=band_width,
        use_seed=seed,
        use_streams=streams,
    )
    elapsed = (time.perf_counter() - t0) * 1000.0

    mode = "streams" if streams else ("seed" if seed else "sw-only")
    return BenchResult(
        name=f"hybaligner ({mode})",
        n_reads=summary["n_reads"],
        read_len=0,
        ref_len=summary["ref_len"],
        elapsed_ms=elapsed,
        throughput_reads_per_sec=summary["throughput_reads_per_sec"],
        n_aligned=summary["n_aligned"],
        score_mean=summary["score_mean"],
        score_max=summary["score_max"],
        extra={
            "tool": "hybaligner",
            "mode": mode,
            "gpu_batches": summary["gpu_batches"],
            "band_width": summary["band_width"],
        },
    )


# ---------------------------------------------------------------------------
# HybAligner FastPipeline runner (zero-overhead fast path)
# ---------------------------------------------------------------------------
_fast_pipeline: 'FastPipeline | None' = None  # persistent across runs

def run_hybaligner_fast(
    fastq_path: str,
    ref_path: str,
    band_width: int = 50,
) -> BenchResult:
    """Run HybAligner via FastPipeline (minimal Python overhead)."""
    from gpu.fast_align import FastPipeline
    global _fast_pipeline

    t0 = time.perf_counter()
    if _fast_pipeline is None:
        _fast_pipeline = FastPipeline()
    summary = _fast_pipeline.run(
        fastq_path, ref_path,
        band_width=band_width,
    )
    elapsed = (time.perf_counter() - t0) * 1000.0

    return BenchResult(
        name="hybaligner (fast)",
        n_reads=summary["n_reads"],
        read_len=0,
        ref_len=summary["ref_len"],
        elapsed_ms=elapsed,
        throughput_reads_per_sec=summary["throughput_reads_per_sec"],
        n_aligned=summary["n_aligned"],
        score_mean=summary["score_mean"],
        score_max=summary["score_max"],
        extra={
            "tool": "hybaligner",
            "mode": "fast",
            "band_width": summary["band_width"],
        },
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(results: List[BenchResult]):
    print()
    print("=" * 72)
    print("  HybAligner vs Minimap2 — Benchmark Report")
    print("=" * 72)
    print()

    # Header
    header = f"{'Tool':<28} {'Reads':>7} {'Time (ms)':>10} {'Reads/s':>10} {'Aligned':>8} {'Mean Score':>11}"
    print(header)
    print("-" * 72)

    baseline_tp = None
    for r in results:
        print(
            f"{r.name:<28} {r.n_reads:>7} {r.elapsed_ms:>10.1f} "
            f"{r.throughput_reads_per_sec:>10.1f} {r.n_aligned:>8} "
            f"{r.score_mean:>11.2f}"
        )
        if r.name == "minimap2":
            baseline_tp = r.throughput_reads_per_sec

    print("-" * 72)

    # Speedup
    if baseline_tp:
        for r in results:
            if r.name != "minimap2" and r.throughput_reads_per_sec > 0:
                speedup = r.throughput_reads_per_sec / baseline_tp
                faster_or_slower = "faster" if speedup > 1 else "slower"
                print(f"  {r.name}: {speedup:.2f}× {faster_or_slower} than minimap2")
        print()

    print("=" * 72)

    # JSON output
    json_results = []
    for r in results:
        d = {
            "name": r.name,
            "n_reads": r.n_reads,
            "elapsed_ms": r.elapsed_ms,
            "throughput_reads_per_sec": r.throughput_reads_per_sec,
            "n_aligned": r.n_aligned,
            "score_mean": r.score_mean,
            "score_max": r.score_max,
            **r.extra,
        }
        json_results.append(d)

    return json_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HybAligner vs Minimap2",
    )
    parser.add_argument(
        "-n", "--reads", type=int, default=1000,
        help="Number of synthetic reads (default: 1000)",
    )
    parser.add_argument(
        "-l", "--read-len", type=int, default=150,
        help="Read length in bp (default: 150)",
    )
    parser.add_argument(
        "-r", "--ref-len", type=int, default=50000,
        help="Reference length in bp (default: 50000)",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=1024,
        help="HybAligner batch size (default: 1024)",
    )
    parser.add_argument(
        "-w", "--band-width", type=int, default=50,
        help="SW band width (default: 50)",
    )
    parser.add_argument(
        "--streams", action="store_true",
        help="Enable stream pipeline for HybAligner",
    )
    parser.add_argument(
        "--no-seed", action="store_true",
        help="Disable seeding in HybAligner",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output JSON path for benchmark results",
    )
    parser.add_argument(
        "--skip-minimap2", action="store_true",
        help="Skip minimap2 baseline (hybaligner only)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Also run optimized FastPipeline (minimal Python overhead)",
    )
    args = parser.parse_args()

    init_logger(format="human")

    print(f"\nGenerating synthetic data: {args.reads} reads × {args.read_len}bp, "
          f"reference {args.ref_len}bp...")

    # Generate data
    ref = generate_reference(args.ref_len)
    reads = generate_fastq(ref, args.reads, args.read_len)

    with tempfile.TemporaryDirectory() as tmpdir:
        fastq_path = os.path.join(tmpdir, "reads.fastq")
        fasta_path = os.path.join(tmpdir, "ref.fasta")

        write_fastq(reads, fastq_path)
        write_fasta(ref, fasta_path)

        print(f"Data written to {tmpdir}")
        print(f"  FASTQ: {fastq_path} ({args.reads} reads)")
        print(f"  FASTA: {fasta_path} ({args.ref_len} bp)")
        print()

        results: List[BenchResult] = []

        # Minimap2 baseline
        if not args.skip_minimap2:
            print("Running minimap2 baseline...")
            try:
                r_mm2 = run_minimap2(fastq_path, fasta_path)
                results.append(r_mm2)
                print(f"  minimap2: {r_mm2.elapsed_ms:.1f} ms, "
                      f"{r_mm2.throughput_reads_per_sec:.1f} reads/s, "
                      f"{r_mm2.n_aligned} aligned")
            except Exception as e:
                print(f"  minimap2 FAILED: {e}")

        # HybAligner
        print("\nRunning HybAligner...")
        try:
            r_hyb = run_hybaligner(
                fastq_path, fasta_path,
                batch_size=args.batch_size,
                band_width=args.band_width,
                streams=args.streams,
                seed=not args.no_seed,
            )
            results.append(r_hyb)
            print(f"  HybAligner: {r_hyb.elapsed_ms:.1f} ms, "
                  f"{r_hyb.throughput_reads_per_sec:.1f} reads/s, "
                  f"{r_hyb.n_aligned} aligned")
        except Exception as e:
            print(f"  HybAligner FAILED: {e}")
            import traceback
            traceback.print_exc()

        # If streams enabled, also run without streams for comparison
        if args.streams:
            print("\nRunning HybAligner (no streams, for comparison)...")
            try:
                r_hyb_sync = run_hybaligner(
                    fastq_path, fasta_path,
                    batch_size=args.batch_size,
                    band_width=args.band_width,
                    streams=False,
                    seed=not args.no_seed,
                )
                results.append(r_hyb_sync)
            except Exception as e:
                print(f"  HybAligner (sync) FAILED: {e}")

        # FastPipeline (optimized, minimal overhead)
        if args.fast:
            print("\nRunning HybAligner FastPipeline (optimized)...")
            try:
                r_fast = run_hybaligner_fast(
                    fastq_path, fasta_path,
                    band_width=args.band_width,
                )
                results.append(r_fast)
                print(f"  FastPipeline: {r_fast.elapsed_ms:.1f} ms, "
                      f"{r_fast.throughput_reads_per_sec:.1f} reads/s, "
                      f"{r_fast.n_aligned} aligned")
            except Exception as e:
                print(f"  FastPipeline FAILED: {e}")
                import traceback
                traceback.print_exc()

        # Report
        json_results = print_report(results)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
