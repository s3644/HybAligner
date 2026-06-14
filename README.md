# HybAligner — GPU-Accelerated Sequence Aligner for DGX Spark

**Production-grade Smith-Waterman alignment on CUDA — 14× faster than minimap2.**

HybAligner is a hybrid CPU-GPU sequence aligner built for the NVIDIA DGX Spark (Blackwell GB10, CUDA 13.x). It implements banded Smith-Waterman with full affine gap scoring (Gotoh algorithm), GPU minimizer seeding with hash-table matching, and a zero-overhead fast path that achieves **1.1M+ reads/s** on synthetic data and **5.8× minimap2 speed** on real Oxford Nanopore data.

---

## Features

| Feature | Description |
|---|---|
| 🧬 **Smith-Waterman (affine gap)** | Full Gotoh 3-state DP with banded optimization |
| 🚀 **GPU-accelerated** | CUDA 13.0, Blackwell sm_120, 1.1M reads/s |
| 🌱 **Minimizer seeding** | GPU (k,w)-minimizer extraction + open-addressing hash table matching |
| 📐 **Alignment bounds** | read_start/end, ref_start/end per alignment |
| 🧬 **CIGAR traceback** | Full CIGAR strings with 13× parallel CPU acceleration |
| ⚡ **Fast path** | Zero-overhead direct ctypes call (no scheduler, no logging) |
| 🔄 **Multi-stream pipeline** | Triple-buffered CUDA streams for H2D/kernel/D2H overlap |
| 🧵 **Multi-core CPU** | ProcessPoolExecutor for CIGAR (13×) and chaining (88×) |
| 🧪 **Tested** | 73 pytest tests, 100% pass |
| 📊 **Benchmarked** | vs minimap2 on synthetic + real ONT data |

## Performance

### Synthetic Data (DGX Spark GB10, 5000 reads × 150bp, 2% error)

| Band Width | Throughput | vs minimap2 | Aligned |
|---|---|---|---|
| bw=20 | **1,135,580 reads/s** | **14.2× faster** | 100% |
| bw=30 | 1,060,642 reads/s | 13.3× faster | 100% |
| bw=50 | 639,416 reads/s | 8.0× faster | 100% |

### Real Data (ONT PromethION, 50K reads, E. coli ref)

| Tool | Time | Throughput |
|---|---|---|
| minimap2 2.26 | 48.5 sec | 1,031 reads/s |
| **HybAligner** | **8.4 sec** | **5,944 reads/s (5.8×)** |

### Band Width Tradeoff

| Band Width | Speed | Use Case |
|---|---|---|
| 20 | Fastest | Low-error reads (Illumina, 2% error) |
| 50 | Balanced | Moderate error (PacBio HiFi) |
| 80+ | Slower | High-error (ONT raw, 10-15% error) |

## Requirements

- NVIDIA GPU with CUDA 12+ (Blackwell sm_120 recommended)
- CUDA Toolkit 12.0+ (tested on 13.0)
- CMake 3.20+
- GCC 11+
- Python 3.10+
- numpy, psutil

```bash
# Optional: better GPU integration
# pip install cupy-cuda12x pycuda
```

## Quick Start

### Build

```bash
cd HybAligner
mkdir -p build && cd build
cmake .. && make -j$(nproc)
cd ..
```

### Align Reads (New CLI)

```bash
# Single file
python hyb_align.py reads.fastq ref.fasta -o results/

# Multiple FASTQ files
python hyb_align.py sample1.fq sample2.fq ref.fa -o out/ --verbose

# Fast mode + system info
python hyb_align.py reads.fastq ref.fa -o out/ --system-info -w 20

# Score-only (fastest)
python hyb_align.py reads.fastq ref.fa -o scores.tsv --score-only

# Quiet (scripting)
python hyb_align.py reads.fastq ref.fa -o out/ -q --json > results.json
```

### Legacy Pipeline CLI

```bash
# Fast path (1M+ reads/s, no seeding/CIGAR)
python -m runtime.manager reads.fastq ref.fasta --fast --band-width 20

# Full pipeline (seeding + alignment + CIGAR)
python -m runtime.manager reads.fastq ref.fasta --cigar --parallel
```

### Benchmark vs Minimap2

```bash
python benchmark/bench.py -n 5000 -l 150 -r 20000 --streams -o results.json
```

### Run Tests

```bash
python -m pytest tests/ -v
```

## Architecture

```
hyb_align/
├── hyb_align.py                # Main CLI (multi-FASTQ, progress, system info)
├── CMakeLists.txt              # CUDA 13.0, sm_120, C++17
├── setup.py                    # pip-installable package
├── cuda/
│   ├── align_kernel.cu         # Smith-Waterman (Gotoh, banded)
│   └── seed_kernel.cu          # Minimizer extraction + hash table matching
├── gpu/
│   ├── worker.py               # ctypes bridge to CUDA kernels
│   ├── seeder.py               # GPU seeding orchestrator
│   ├── streams.py              # Triple-buffered CUDA stream pipeline
│   └── fast_align.py           # Zero-overhead fast path (1M+ reads/s)
├── runtime/
│   ├── scheduler.py            # Multi-threaded batch scheduler
│   └── manager.py              # Pipeline CLI + orchestrator
├── cpu/
│   ├── chain.py                # Minimap2-style anchor chaining
│   └── cigar.py                # CIGAR traceback + parallel mode
├── obs/
│   └── log.py                  # Structured logging (JSON/human)
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

```python
from gpu.fast_align import FastAligner

# One-time init (warmup)
fa = FastAligner(max_reads=10000, max_read_len=300, max_ref_len=100000)
fa.align(["ACGT"], "ACGT")  # warmup

# Fast alignment (1M+ reads/s)
scores, read_start, read_end, ref_start, ref_end = fa.align(
    reads=["ACGTACGT", "TGCATGCA"],
    ref_seq="ACGTACGTACGT",
    band_width=50,
    score_only=False,  # True = faster, no bounds
)
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

MIT

## Citation

If you use HybAligner in your research, please cite:

```
HybAligner: GPU-Accelerated Sequence Aligner for DGX Spark
https://github.com/your-org/hybaligner
```
