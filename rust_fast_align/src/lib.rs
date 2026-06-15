//! HybAlign — GPU-accelerated sequence aligner (Rust fast path).
//!
//! Provides a zero-overhead alignment pipeline:
//! 1. Memory-map FASTQ file
//! 2. Read FASTA reference
//! 3. Encode reads into flat padded buffer
//! 4. Dispatch to CUDA kernel
//! 5. Return results
//!
//! ```ignore
//! use hyb_align::{FastqFile, FastAligner, encode_from_fastq, encode_reference, AlignParams};
//!
//! let fq = FastqFile::open("reads.fastq")?;
//! let ref_seq = std::fs::read_to_string("ref.fasta")?;
//! let ref_bytes = encode_reference(&ref_seq);
//! let read_bytes = encode_from_fastq(&fq, fq.max_len());
//!
//! let mut fa = FastAligner::new(fq.n_reads() + 10, fq.max_len(), ref_bytes.len() + 100)?;
//! let result = fa.align(&read_bytes, &ref_bytes, fq.n_reads(), fq.max_len(),
//!                       ref_bytes.len(), AlignParams::default())?;
//! println!("{} reads aligned", result.scores.iter().filter(|&&s| s > 0.0).count());
//! ```

pub mod align;
pub mod cuda;
pub mod encode;
pub mod fastq;
pub mod seed;

// Re-exports
pub use align::{AlignParams, FastAligner};
pub use cuda::{AlignResult, CudaKernel};
pub use encode::{encode_from_fastq, encode_reads, encode_reference, encode_reference_bytes};
pub use fastq::FastqFile;
pub use seed::{best_anchor, Anchor, SeedIndex};
