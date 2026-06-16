//! CUDA FFI bindings — wraps libcuda_kernels.so via libloading.
//!
//! Maps launch_sw_affine directly: Python ctypes → Rust FFI.
//! One call aligns all reads against the reference on GPU.

use std::ffi::c_int;

/// Result of a single SW affine alignment batch.
#[derive(Debug, Clone)]
pub struct AlignResult {
    pub scores: Vec<f32>,
    pub read_start: Vec<i32>,
    pub read_end: Vec<i32>,
    pub ref_start: Vec<i32>,
    pub ref_end: Vec<i32>,
}

/// CUDA kernel launcher wrapping the compiled shared library.
///
/// ```ignore
/// let kernel = CudaKernel::load(None)?;
/// let result = kernel.launch_sw_affine(
///     &reads_bytes, &ref_bytes,
///     1000, 150, 50000,  // n_reads, read_len, ref_len
///     50, 5, 2, 256,      // band, gap_open, gap_extend, block_size
/// )?;
/// ```
pub struct CudaKernel {
    // Function pointer extracted from the leaked library.
    // We intentionally leak the Library handle so the function pointer
    // remains valid for the process lifetime. OS cleans up on exit.
    launch_sw_affine: unsafe extern "C" fn(
        reads: *const u8,
        ref_seq: *const u8,
        scores: *mut f32,
        read_start: *mut c_int,
        read_end: *mut c_int,
        ref_start: *mut c_int,
        ref_end: *mut c_int,
        num_reads: c_int,
        read_len: c_int,
        ref_len: c_int,
        band_width: c_int,
        gap_open: c_int,
        gap_extend: c_int,
        block_size: c_int,
    ) -> c_int,
}

impl CudaKernel {
    /// Load libcuda_kernels.so from standard locations.
    ///
    /// Search order:
    /// 1. `lib_path` if provided
    /// 2. `./build/libcuda_kernels.so`
    /// 3. `../build/libcuda_kernels.so` (relative to rust_fast_align/)
    pub fn load(lib_path: Option<&str>) -> anyhow::Result<Self> {
        let path = if let Some(p) = lib_path {
            p.to_string()
        } else {
            let candidates = [
                "./build/libcuda_kernels.so",
                "../build/libcuda_kernels.so",
                "../../build/libcuda_kernels.so",
            ];
            let mut found = None;
            for c in &candidates {
                if std::path::Path::new(c).exists() {
                    found = Some(c.to_string());
                    break;
                }
            }
            found.ok_or_else(|| {
                anyhow::anyhow!(
                    "libcuda_kernels.so not found. Searched: {:?}. \
                     Run 'make' in the HybAligner root first.",
                    candidates
                )
            })?
        };

        unsafe {
            let lib = libloading::Library::new(&path)?;
            let func: libloading::Symbol<
                unsafe extern "C" fn(
                    *const u8, *const u8, *mut f32,
                    *mut c_int, *mut c_int, *mut c_int, *mut c_int,
                    c_int, c_int, c_int,
                    c_int, c_int, c_int, c_int,
                ) -> c_int,
            > = lib.get(b"launch_sw_affine")?;

            // Extract raw function pointer and leak the library
            // so the symbol stays valid for 'static lifetime.
            let func_ptr: unsafe extern "C" fn(
                *const u8, *const u8, *mut f32,
                *mut c_int, *mut c_int, *mut c_int, *mut c_int,
                c_int, c_int, c_int,
                c_int, c_int, c_int, c_int,
            ) -> c_int = *func;
            std::mem::forget(lib);

            Ok(Self {
                launch_sw_affine: func_ptr,
            })
        }
    }

    /// Run Smith-Waterman affine-gap alignment on the GPU.
    ///
    /// All reads and the reference must already be encoded as flat bytes
    /// (each read padded/truncated to exactly `read_len` with 'N').
    pub fn launch_sw_affine(
        &self,
        reads_bytes: &[u8],
        ref_bytes: &[u8],
        n_reads: usize,
        read_len: usize,
        ref_len: usize,
        band_width: i32,
        gap_open: i32,
        gap_extend: i32,
        block_size: i32,
    ) -> anyhow::Result<AlignResult> {
        let n = n_reads as c_int;
        let rl = read_len as c_int;
        let rfl = ref_len as c_int;

        let mut scores = vec![0.0f32; n_reads];
        let mut read_start = vec![0i32; n_reads];
        let mut read_end = vec![0i32; n_reads];
        let mut ref_start = vec![0i32; n_reads];
        let mut ref_end = vec![0i32; n_reads];

        let ret = unsafe {
            (self.launch_sw_affine)(
                reads_bytes.as_ptr(),
                ref_bytes.as_ptr(),
                scores.as_mut_ptr(),
                read_start.as_mut_ptr(),
                read_end.as_mut_ptr(),
                ref_start.as_mut_ptr(),
                ref_end.as_mut_ptr(),
                n, rl, rfl,
                band_width, gap_open, gap_extend, block_size,
            )
        };

        if ret != 0 {
            anyhow::bail!("SW affine kernel failed with code {}", ret);
        }

        Ok(AlignResult {
            scores,
            read_start,
            read_end,
            ref_start,
            ref_end,
        })
    }
}

// SAFETY: CudaKernel owns the library handle. The extern "C" functions
// are thread-safe (CUDA streams serialize GPU access).
unsafe impl Send for CudaKernel {}
unsafe impl Sync for CudaKernel {}

// ═══════════════════════════════════════════════════════════
// Seed kernel bindings (extract_minimizers, hash table, match)
// ═══════════════════════════════════════════════════════════

/// GPU seeding kernel — extracts minimizers, builds hash table, matches seeds.
pub struct SeedKernel {
    extract: unsafe extern "C" fn(
        seq: *const u8,
        hashes: *mut u64,
        positions: *mut c_int,
        counts: *mut c_int,
        num_seqs: c_int, seq_len: c_int,
        k: c_int, w: c_int, max_mins: c_int,
        block_size: c_int,
    ) -> c_int,
    build_ht: unsafe extern "C" fn(
        ref_hashes: *const u64, ref_pos: *const c_int,
        n_ref_mins: c_int,
        table_keys: *mut u64, table_vals: *mut c_int,
        table_size: c_int, max_vals_per_key: c_int,
        block_size: c_int,
    ) -> c_int,
    match_ht: unsafe extern "C" fn(
        read_hashes: *const u64, read_pos: *const c_int,
        table_keys: *const u64, table_vals: *const c_int,
        table_size: c_int, max_vals_per_key: c_int,
        anchor_rp: *mut c_int, anchor_fp: *mut c_int,
        anchor_counts: *mut c_int,
        num_reads: c_int, max_mins: c_int, max_anchors: c_int,
        block_size: c_int,
    ) -> c_int,
}

impl SeedKernel {
    pub fn load(lib_path: Option<&str>) -> anyhow::Result<Self> {
        let path = if let Some(p) = lib_path { p.to_string() } else {
            for c in &["./build/libcuda_kernels.so", "../build/libcuda_kernels.so", "../../build/libcuda_kernels.so"] {
                if std::path::Path::new(c).exists() { return Self::load(Some(c)); }
            }
            anyhow::bail!("libcuda_kernels.so not found")
        };

        unsafe {
            let lib = libloading::Library::new(&path)?;
            let extract: libloading::Symbol<unsafe extern "C" fn(*const u8, *mut u64, *mut c_int, *mut c_int, c_int, c_int, c_int, c_int, c_int, c_int) -> c_int> = lib.get(b"launch_extract_minimizers")?;
            let build_ht: libloading::Symbol<unsafe extern "C" fn(*const u64, *const c_int, c_int, *mut u64, *mut c_int, c_int, c_int, c_int) -> c_int> = lib.get(b"launch_build_hash_table")?;
            let match_ht: libloading::Symbol<unsafe extern "C" fn(*const u64, *const c_int, *const u64, *const c_int, c_int, c_int, *mut c_int, *mut c_int, *mut c_int, c_int, c_int, c_int, c_int) -> c_int> = lib.get(b"launch_match_hash_table")?;
            let extract = *extract;
            let build_ht = *build_ht;
            let match_ht = *match_ht;
            std::mem::forget(lib);
            Ok(Self { extract, build_ht, match_ht })
        }
    }

    pub fn build_ref_index(&self, ref_bytes: &[u8], k: i32, w: i32) -> anyhow::Result<GpuSeedIndex> {
        let ref_len = ref_bytes.len() as i32;
        let max_mins = 10000i32.min(ref_len / (k + w) + 10);
        let n_ref = 1i32;

        let mut hashes = vec![0u64; (n_ref * max_mins) as usize];
        let mut positions = vec![0i32; (n_ref * max_mins) as usize];
        let mut counts = vec![0i32; n_ref as usize];

        unsafe { (self.extract)(ref_bytes.as_ptr(), hashes.as_mut_ptr(), positions.as_mut_ptr(), counts.as_mut_ptr(), n_ref, ref_len, k, w, max_mins, 256); }
        let n_mins = counts[0] as usize;
        if n_mins == 0 { return Ok(GpuSeedIndex { table_keys: vec![], table_vals: vec![], table_size: 0, n_mins: 0 }); }

        let mut table_size = 1usize;
        while table_size < n_mins * 2 { table_size *= 2; }
        let mut table_keys = vec![0xFFFFFFFFFFFFFFFFu64; table_size];
        let max_vals = 8i32;
        let mut table_vals = vec![-1i32; table_size * max_vals as usize];

        unsafe { (self.build_ht)(hashes.as_ptr(), positions.as_ptr(), n_mins as i32, table_keys.as_mut_ptr(), table_vals.as_mut_ptr(), table_size as i32, max_vals, 256); }
        Ok(GpuSeedIndex { table_keys, table_vals, table_size, n_mins })
    }

    pub fn seed_batch(&self, reads_bytes: &[u8], n_reads: i32, read_len: i32, idx: &GpuSeedIndex, k: i32, w: i32) -> anyhow::Result<(Vec<i32>, Vec<i32>)> {
        let max_mins = 100i32.min(read_len / (k + w) + 5);
        let mut rh = vec![0u64; (n_reads * max_mins) as usize];
        let mut rp = vec![0i32; (n_reads * max_mins) as usize];
        let mut rc = vec![0i32; n_reads as usize];
        unsafe { (self.extract)(reads_bytes.as_ptr(), rh.as_mut_ptr(), rp.as_mut_ptr(), rc.as_mut_ptr(), n_reads, read_len, k, w, max_mins, 256); }

        let max_anchors = 16i32;
        let mut arp = vec![0i32; (n_reads * max_anchors) as usize];
        let mut afp = vec![0i32; (n_reads * max_anchors) as usize];
        let mut ac = vec![0i32; n_reads as usize];
        unsafe { (self.match_ht)(rh.as_ptr(), rp.as_ptr(), idx.table_keys.as_ptr(), idx.table_vals.as_ptr(), idx.table_size as i32, 8, arp.as_mut_ptr(), afp.as_mut_ptr(), ac.as_mut_ptr(), n_reads, max_mins, max_anchors, 256); }

        let mut best_rp = vec![-1i32; n_reads as usize];
        let mut best_fp = vec![-1i32; n_reads as usize];
        for i in 0..n_reads as usize {
            if ac[i] > 0 {
                best_rp[i] = arp[i * max_anchors as usize];
                best_fp[i] = afp[i * max_anchors as usize];
            }
        }
        Ok((best_rp, best_fp))
    }
}

pub struct GpuSeedIndex {
    pub table_keys: Vec<u64>,
    pub table_vals: Vec<i32>,
    pub table_size: usize,
    pub n_mins: usize,
}
