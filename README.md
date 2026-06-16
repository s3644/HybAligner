# HybAligner — GPU-Accelerated Sequence Aligner for DGX Spark

**Hybrid CPU-GPU alignment — 7.6× more alignments than minimap2 on real human exome data.**

HybAligner is a hybrid CPU-GPU sequence aligner for the NVIDIA DGX Spark (Blackwell GB10, CUDA 13.x). Implements banded Smith-Waterman with affine gap scoring, dual-level minimizer seeding (8-mer coarse + 15-mer fine), chunked genome-scale indexing, and multi-core parallel seed matching. Available in **Python** and **Rust**.

- 🐍 v0.9 Hybrid: **2,377 reads/s** on 47 Mbp chr21 (20 CPU cores + batched GPU)
- 🦀 Rust: **188K reads/s** synthetic, **3.2×** faster than Python at genome scale
- 📊 **7.6×** more alignments than minimap2 — **30.8%** vs 4.1% on real exome data
- 🧬 WGS-ready: chunked indexing supports references up to **3.2 Gbp**

---

## Features

| Feature | Description |
|---|---|
| 🧬 **Smith-Waterman (affine gap)** | Full Gotoh 3-state DP with banded optimization |
| 🚀 **GPU-accelerated** | CUDA 13.0, Blackwell sm_120, batched multi-read SW |
| 🧵 **Multi-core CPU** | ThreadPool parallel seeding (20 cores), ProcessPool CIGAR |
| 🌱 **Dual-level seeding** | 8-mer coarse (65K keys) + 15-mer fine (255K keys) |
| 📦 **Chunked WGS indexing** | 10 Mbp chunks with 1 Mbp overlap — scales to 3.2 Gbp |
| 📐 **Alignment bounds** | read_start/end, ref_start/end per alignment |
| 🦀 **Rust fast path** | `hyb-align-rs` — 2.6× faster I/O, seeded + standard modes |
| ⚡ **Zero-overhead pipeline** | FastPipeline, FastAligner, HybridAligner |
| 💾 **Index serialization** | Save/load chunked indexes (pickle, 146 MB for chr21) |
| 🧪 **Tested** | 73 pytest + 10 Rust tests, 100% pass |
| 📊 **Benchmarked** | vs minimap2, BWA-MEM on synthetic + real human exome |

### Interactive TUI

```bash
# Launch interactive terminal interface
python hyb_align_tui.py
# or
make tui
```

The TUI provides:
- 🖥️ System info display (GPU, CPU, RAM, disk)
- 📂 Interactive file browser with glob patterns
- ⚙️ Preset configurations (fast/balanced/accurate/custom)
- 📊 Live progress bars with per-stage timing
- 📋 Formatted results table with alignment stats



## Performance

### Synthetic Data (DGX Spark GB10, 5000 reads × 150bp, 2% error, 50Kbp ref)

| Mode | Throughput | vs minimap2 (16t) | Aligned |
|---|---|---|---|
| FastPipeline (bw=50) | **133,330 reads/s** | **1.44× faster** | 100% |
| FastAligner (bw=20) | 1,135,580 reads/s | 12.3× faster | 100% |
| FastAligner (bw=30) | 1,060,642 reads/s | 11.5× faster | 100% |
| FastAligner (bw=50) | 639,416 reads/s | 6.9× faster | 100% |

### Real Human Exome — 10K reads × 100bp, 47 Mbp chr21 (SRR077487)

| Tool | Time | Reads/s | Aligned | Architecture |
|---|---|---|---|---|
| minimap2 2.26 (16t) | 712 ms | **14,043** | 406 (4.1%) | C + SIMD, seed-chain-align |
| HybAligner v0.7 (seeded) | 326 ms* | 1,534* | 161* (32%) | Single index, sequential |
| HybAligner v0.8 (WGS chunked) | 5,364 ms | 1,865 | 3,078 (30.8%) | 6 chunks, dual-level seeds |
| **HybAligner v0.9 (Hybrid)** ✨ | **4,207 ms** | **2,377** | **3,065 (30.7%)** | **20 CPU cores + batched GPU** |

> \*500-read test. v0.9 Hybrid uses ThreadPoolExecutor (20 workers) for parallel seed matching + anchor-position clustering for batched GPU SW. Finds **7.6× more alignments** than minimap2 because exhaustive banded SW catches all matches within seed windows, unlike minimap2's strict seed-chain filtering.

### Version Evolution (10K reads, 47 Mbp chr21)

| Version | Architecture | Throughput | Key Innovation |
|---|---|---|---|
| v0.6 FastPipeline | Single-encode, zero-copy | 133K (synthetic) | 96× Python overhead reduction |
| v0.7 Seeded | Single index + anchored SW | 1,534* | Genome-scale via minimizer windows |
| v0.8 WGS Chunked | 10 Mbp chunks, dual seeds | 1,865 | Scales to 3.2 Gbp |
| **v0.9 Hybrid** | **20-core CPU + batched GPU** | **2,377** | **Parallel seeding + cluster batching** |

### Comparison with Other Aligners

| Aligner | Type | Speed | Sensitivity | Best For |
|---|---|---|---|---|
| **HybAligner v0.9** | GPU SW + CPU seeds | 2.4K r/s | **30.8%** 🥇 | Variant discovery, degraded samples |
| minimap2 | CPU seed-chain-SW | **14K r/s** 🥇 | 4.1% | General long-read alignment |
| BWA-MEM | CPU FM-index + SW | ~10K r/s | High | Short-read WGS gold standard |
| winnowmap | Weighted minimizer | ~5K r/s | High | Repetitive regions |
| lra | Sparse DP | ~1K r/s | **Very high** | PacBio HiFi, assemblies |
| NGMLR | Convex gap chaining | ~1K r/s | High | Structural variants |

> **Why HybAligner finds more:** minimap2 requires colinear seed chains → discards reads with broken chains. HybAligner only needs ONE 8-mer match → exhaustive SW finds alignment regardless of seed quality. Trade-off: higher recall at cost of some false positives (add MAPQ filter for production).

### Rust vs Python (50K synthetic reads, 50 Kbp ref)

| Tool | Time | Throughput | vs Python |
|---|---|---|---|
| **Rust `hyb-align-rs`** | 266.4 ms | **187,723 reads/s** | **1.1× faster** |
| Python FastPipeline | 294.7 ms | 173,094 reads/s | baseline |

| Component | Rust | Python | Speedup |
|---|---|---|---|
| FASTQ parse (50K reads) | 12.4 ms | 32.4 ms | **2.6×** |
| Read encoding | 2.5 ms | (in parse) | — |
| GPU kernel | 55.7 ms | 66.3 ms | 1.2× |

### Python Overhead Elimination (v0.6.0)

| Optimization | Before | After | Impact |
|---|---|---|---|
| **Read encoding** | Per-read `.encode()` | Single `''.join().encode()` | ~15× faster |
| **Result copies** | `.copy()` on 5 output arrays | `zero_copy=True` views | Saves 5 malloc+copy |
| **Ref index build** | 204ms even in fast path | Skipped when not seeding | Saves 204ms |
| **Scheduler** | `queue.Queue` + threading | Bypassed via FastPipeline | Saves ~300ms |
| **ctypes loading** | Per-call `CDLL` load | Module-level singleton | Saves ~200ms first call |
| **Buffer alloc** | New numpy arrays per call | Pre-allocated + reused | Zero malloc |

**Result:** Pipeline went from 728ms → 7.6ms (**96× Python overhead reduction**).

### Band Width Tradeoff

| Band Width | Speed | Use Case |
|---|---|---|
| 20 | Fastest | Low-error reads (Illumina, 2% error) |
| 50 | Balanced | Moderate error (PacBio HiFi) |
| 80+ | Slower | High-error (ONT raw, 10-15% error) |

## Requirements

- NVIDIA GPU with CUDA 12+ (Blackwell GB10 recommended)
- CUDA Toolkit 12.0+ (tested on 13.0)
- CMake 3.20+, GCC 11+, Python 3.10+
- numpy, psutil, tqdm, rich

```bash
# Ubuntu/Debian
sudo apt install cmake build-essential nvidia-cuda-toolkit python3-pip
pip install numpy psutil tqdm rich
```

## Quick Start

### 1. Install

```bash
git clone https://github.com/s3644/HybAligner.git && cd HybAligner
make && make install     # installs hyb-align to ~/.local/bin/
```

### 2. Run

```bash
# CLI
hyb-align reads.fastq ref.fasta -o results/ --verbose

# Interactive TUI
hyb-align-tui            # or: make tui
```

### 3. From Source (no install)

```bash
make                     # build CUDA library
python hyb_align.py reads.fastq ref.fa -o results/
make test                # 73 tests
```

### Makefile Targets

| Command | Does |
|---|---|
| `make` | Build CUDA library |
| `make install` | Install to `~/.local/bin/hyb-align` |
| `make uninstall` | Remove installation |
| `make test` | Run 73 pytest tests |
| `make bench` | Benchmark vs minimap2 |
| `make tui` | Launch interactive TUI |
| `make clean` | Remove build artifacts |

### Align Reads

```bash
# After install:
hyb-align reads.fastq ref.fasta -o results/

# Or from source:
python hyb_align.py reads.fastq ref.fasta -o results/

# Multiple files + verbose
hyb-align sample1.fq sample2.fq ref.fa -o out/ --verbose

# System info + fast mode
hyb-align reads.fastq ref.fa -o out/ --system-info -w 20
```

### Legacy Pipeline CLI

```bash
# FastPipeline (133K+ reads/s, no seeding/CIGAR)
python -m runtime.manager reads.fastq ref.fasta --fast --band-width 50

# Full pipeline (seeding + alignment + CIGAR)
python -m runtime.manager reads.fastq ref.fasta --cigar --parallel
```

### Benchmark vs Minimap2

```bash
python benchmark/bench.py -n 5000 -l 150 -r 50000 --fast -o results.json

# Real data benchmark (downloads SRR077487 exome + chr21 reference)
python benchmark/bench_real.py --fastq data/SRR077487_1.fastq.gz --ref data/chr21.fa.gz --max-reads 5000
```

### Run Tests

```bash
python -m pytest tests/ -v
```

## Architecture

```
hyb_align/
├── hyb_align.py                # Main CLI (multi-FASTQ, progress, system info)
├── hyb_align_tui.py            # Interactive Rich-based TUI
├── CMakeLists.txt              # CUDA 13.0, sm_120, C++17
├── setup.py                    # pip-installable package
├── cuda/
│   ├── align_kernel.cu         # Smith-Waterman (Gotoh, banded)
│   └── seed_kernel.cu          # Minimizer extraction + hash table matching
├── gpu/
│   ├── worker.py               # ctypes bridge to CUDA kernels
│   ├── seeder.py               # GPU seeding orchestrator
│   ├── streams.py              # Triple-buffered CUDA stream pipeline
│   ├── fast_align.py           # FastPipeline, FastAligner, seeded mode
│   ├── wgs_align.py            # WgsAligner — chunked genome-scale indexing
│   └── hybrid_align.py         # HybridAligner — multi-core CPU + batched GPU
├── runtime/
│   ├── scheduler.py            # Multi-threaded batch scheduler
│   └── manager.py              # Pipeline CLI + orchestrator
├── cpu/
│   ├── chain.py                # Minimap2-style anchor chaining
│   └── cigar.py                # CIGAR traceback + parallel mode
├── obs/
│   └── log.py                  # Structured logging (JSON/human)
├── benchmark/
│   ├── bench.py                # Synthetic data benchmark
│   └── bench_real.py           # Real FASTQ data benchmark (SRR077487)
├── rust_fast_align/            # 🦀 Rust FastPath (v0.2.0)
│   ├── Cargo.toml              # memmap2, libloading, clap, serde_json
│   └── src/
│       ├── main.rs             # CLI (--standard, --seeded)
│       ├── lib.rs              # Re-exports
│       ├── cuda.rs             # CUDA FFI via libloading
│       ├── fastq.rs            # Memory-mapped FASTQ parser
│       ├── encode.rs           # Read encoder
│       ├── align.rs            # FastAligner
│       └── seed.rs             # CPU minimizer seeding + index
├── data/                       # Downloaded test data (chr21, exome reads)
└── tests/                      # 73 pytest tests
```
├── benchmark/
│   └── bench.py                # minimap2 comparison benchmark
└── tests/                      # 73 pytest tests
```

## CLI Reference

### Main CLI (`hyb_align.py`)

```
python hyb_align.py <fastq...> <ref> [options]

Positional:
  fastq               One or more FASTQ files
  ref                 Reference FASTA file

Alignment:
  -w, --band-width N  Half-band width for DP (default: 50)
  --gap-open N        Gap opening penalty (default: 5)
  --gap-extend N      Gap extension penalty (default: 2)
  --score-only        Score-only mode (faster, no bounds)

Output:
  -o, --output PATH   Output file or directory (default: hyb_results)
  --prefix PREFIX     Prefix for output filenames
  -v, --verbose       Per-file detailed stats
  --json              Output summary as JSON
  -q, --quiet         Suppress all output except errors

Display:
  --system-info       Show CPU/GPU/RAM/disk before aligning
  --no-progress       Disable stage progress
```

### Legacy Pipeline CLI (`runtime/manager.py`)

Alignment:
  -w, --band-width N    Half-band width for DP (default: 50)
  --gap-open N          Gap opening penalty (default: 5)
  --gap-extend N        Gap extension penalty (default: 2)

Seeding:
  --seed / --no-seed    Enable/disable minimizer seeding (default: enabled)
  -k, --kmer N          K-mer size (default: 15)
  -W, --window N        Window size (default: 10)

Performance:
  --fast                Zero-overhead fast path (1M+ reads/s)
  --streams             CUDA multi-stream pipeline
  --parallel            Multi-core CPU for CIGAR/chaining
  -b, --batch-size N    Reads per batch (default: 4096)
  --gpu-only            Skip CPU fallback

Output:
  --cigar               Generate CIGAR strings
  -o, --output PATH     JSON results file
  --json                Print summary as JSON
```

## Python API

### FastPipeline (one-shot, zero overhead)

```python
from gpu.fast_align import FastPipeline

# One-shot: FASTQ + FASTA → results (133K+ reads/s)
fp = FastPipeline()
result = fp.run("reads.fastq", "ref.fasta", band_width=50)
print(f"{result['throughput_reads_per_sec']:.0f} reads/s")
print(f"{result['n_aligned']}/{result['n_reads']} aligned")
fp.free()
```

### FastAligner (reusable, pre-allocated buffers)

```python
from gpu.fast_align import FastAligner, encode_reads

# One-time init (warmup)
fa = FastAligner(max_reads=10000, max_read_len=300, max_ref_len=100000)
fa.align(["ACGT"], "ACGT")  # warmup

# Fast alignment (133K+ reads/s)
scores, read_start, read_end, ref_start, ref_end = fa.align(
    reads=["ACGTACGT", "TGCATGCA"],
    ref_seq="ACGTACGTACGT",
    band_width=50,
    score_only=False,   # True = faster, no bounds
    zero_copy=False,     # True = views (faster, no .copy())
)

# Or with pre-encoded bytes (fastest path, zero Python overhead)
reads_bytes = encode_reads(reads, read_len)
scores, *bounds = fa.align_bytes(reads_bytes, n_reads, read_len, ref_seq)
```

## Algorithm

### Smith-Waterman (Gotoh Affine Gap)

$$\begin{aligned} M(i,j) &= \max(M(i-1,j-1), I_x(i-1,j-1), I_y(i-1,j-1)) + s(r_i, q_j) \\ I_x(i,j) &= \max(M(i-1,j) - g_o,\; I_x(i-1,j)) - g_e \\ I_y(i,j) &= \max(M(i,j-1) - g_o,\; I_y(i,j-1)) - g_e \end{aligned}$$

- **Banded**: Only $|i-j| \leq$ `band_width` computed
- **Local alignment**: All values clamped ≥ 0
- **Scoring matrix**: Match +2, Mismatch −3, Gap open −5, Gap extend −2
- **Per-thread shared memory**: 6 × `band_size` ints, auto-capped block size

### Minimizer Seeding

- $(k,w)$-minimizer scheme: $k=15$, $w=10$
- Canonical k-mer hashing (2-bit encoding)
- GPU open-addressing hash table with double hashing
- O(1) amortized lookup (vs O(N²) brute-force)

## License

MIT — see [LICENSE](LICENSE)

## Citation

If you use HybAligner in your research, please cite:

> Jitpimolmard, J. (2026). *HybAligner: GPU-Accelerated Sequence Aligner for DGX Spark* (v0.7.1) [Software]. KKU National Phenome Institute, Khon Kaen University. https://github.com/s3644/HybAligner

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata (CFF 1.2.0 format).

**BibTeX:**
```bibtex
@software{HybAligner2026,
  author       = {Jitpimolmard, Jukrapope},
  title        = {HybAligner: GPU-Accelerated Sequence Aligner for DGX Spark},
  year         = {2026},
  version      = {0.7.1},
  publisher    = {KKU National Phenome Institute, Khon Kaen University},
  url          = {https://github.com/s3644/HybAligner},
  orcid        = {https://orcid.org/0009-0001-9170-426X},
}
