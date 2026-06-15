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

