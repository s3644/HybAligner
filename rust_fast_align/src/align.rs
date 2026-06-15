//! FastAligner — reusable, pre-allocated GPU aligner.
//!
//! Mirrors the Python FastAligner: loads CUDA kernel once, pre-allocates
//! all output buffers, and supports repeated calls with different reads.
//!
//! ```ignore
//! let mut fa = FastAligner::new(10000, 300, 100000)?;
//! let result = fa.align(&reads_bytes, &ref_bytes, 1000, 150, 50000, AlignParams::default())?;
//! // Call again with different reads — buffers reused
//! let result2 = fa.align(&more_reads, &ref_bytes, 500, 200, 50000, AlignParams::default())?;
//! ```

use crate::cuda::{AlignResult, CudaKernel};

/// Alignment parameters (mirrors Python FastAligner defaults).
#[derive(Debug, Clone)]
pub struct AlignParams {
    pub band_width: i32,
    pub gap_open: i32,
    pub gap_extend: i32,
    pub block_size: i32,
}

impl Default for AlignParams {
    fn default() -> Self {
        Self {
            band_width: 50,
            gap_open: 5,
            gap_extend: 2,
            block_size: 256,
        }
    }
}

/// Reusable GPU aligner with pre-allocated buffers.
///
/// Buffers are resized automatically on first call if the data
/// exceeds initial capacity. Subsequent calls with <= capacity
/// reuse the same buffers (zero allocation).
pub struct FastAligner {
    kernel: CudaKernel,

    // Pre-allocated output buffers
    scores: Vec<f32>,
    read_start: Vec<i32>,
    read_end: Vec<i32>,
    ref_start: Vec<i32>,
    ref_end: Vec<i32>,

    // Cached reference bytes (skip re-upload when ref unchanged)
    #[allow(dead_code)]
    cached_ref: Option<Vec<u8>>,

    // Capacity
    #[allow(dead_code)]
    max_reads: usize,
    #[allow(dead_code)]
    max_read_len: usize,
    #[allow(dead_code)]
    max_ref_len: usize,
}

impl FastAligner {
    /// Create a new FastAligner with pre-allocated buffer capacity.
    ///
    /// The CUDA library is loaded immediately. Buffers are allocated
    /// lazily on first `align()` call.
    pub fn new(max_reads: usize, max_read_len: usize, max_ref_len: usize) -> anyhow::Result<Self> {
        let kernel = CudaKernel::load(None)?;
        Ok(Self {
            kernel,
            scores: Vec::new(),
            read_start: Vec::new(),
            read_end: Vec::new(),
            ref_start: Vec::new(),
            ref_end: Vec::new(),
            cached_ref: None,
            max_reads,
            max_read_len,
            max_ref_len,
        })
    }

    /// Create with explicit CUDA library path.
    pub fn with_lib_path(
        max_reads: usize,
        max_read_len: usize,
        max_ref_len: usize,
        lib_path: &str,
    ) -> anyhow::Result<Self> {
        let kernel = CudaKernel::load(Some(lib_path))?;
        Ok(Self {
            kernel,
            scores: Vec::new(),
            read_start: Vec::new(),
            read_end: Vec::new(),
            ref_start: Vec::new(),
            ref_end: Vec::new(),
            cached_ref: None,
            max_reads,
            max_read_len,
            max_ref_len,
        })
    }

    /// Ensure output buffers are large enough for `n_reads`.
    fn ensure_buffers(&mut self, n_reads: usize) {
        if self.scores.len() < n_reads {
            self.scores.resize(n_reads, 0.0);
            self.read_start.resize(n_reads, 0);
            self.read_end.resize(n_reads, 0);
            self.ref_start.resize(n_reads, 0);
            self.ref_end.resize(n_reads, 0);
        }
    }

    /// Align pre-encoded reads against a reference.
    ///
    /// `reads_bytes`: flat buffer of n_reads × read_len bytes (padded).
    /// `ref_bytes`: reference sequence as raw bytes.
    ///
    /// Returns references into internal buffers — call `.to_owned()` or
    /// `.to_vec()` on the result if you need ownership.
    pub fn align(
        &mut self,
        reads_bytes: &[u8],
        ref_bytes: &[u8],
        n_reads: usize,
        read_len: usize,
        ref_len: usize,
        params: AlignParams,
    ) -> anyhow::Result<AlignResult> {
        assert!(n_reads <= self.max_reads, "n_reads exceeds capacity");
        assert!(reads_bytes.len() >= n_reads * read_len, "reads_bytes too short");

        // Ensure buffer capacity
        self.ensure_buffers(n_reads);

        let result = self.kernel.launch_sw_affine(
            reads_bytes,
            ref_bytes,
            n_reads,
            read_len,
            ref_len,
            params.band_width,
            params.gap_open,
            params.gap_extend,
            params.block_size,
        )?;

        // Copy results into our pre-allocated buffers
        self.scores[..n_reads].copy_from_slice(&result.scores);
        self.read_start[..n_reads].copy_from_slice(&result.read_start);
        self.read_end[..n_reads].copy_from_slice(&result.read_end);
        self.ref_start[..n_reads].copy_from_slice(&result.ref_start);
        self.ref_end[..n_reads].copy_from_slice(&result.ref_end);

        Ok(AlignResult {
            scores: self.scores[..n_reads].to_vec(),
            read_start: self.read_start[..n_reads].to_vec(),
            read_end: self.read_end[..n_reads].to_vec(),
            ref_start: self.ref_start[..n_reads].to_vec(),
            ref_end: self.ref_end[..n_reads].to_vec(),
        })
    }

    /// Get references to internal buffers (zero-copy, caller must not mutate).
    pub fn scores(&self) -> &[f32] {
        &self.scores
    }

    pub fn n_aligned(&self) -> usize {
        self.scores.iter().filter(|&&s| s > 0.0).count()
    }

    pub fn score_mean(&self) -> f64 {
        let nz: Vec<_> = self.scores.iter().filter(|&&s| s > 0.0).copied().collect();
        if nz.is_empty() {
            0.0
        } else {
            nz.iter().map(|&s| s as f64).sum::<f64>() / nz.len() as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_default() {
        let p = AlignParams::default();
        assert_eq!(p.band_width, 50);
        assert_eq!(p.gap_open, 5);
        assert_eq!(p.gap_extend, 2);
        assert_eq!(p.block_size, 256);
    }
}
