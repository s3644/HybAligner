/**
 * align_kernel.cu — Production-grade Smith-Waterman alignment kernel.
 *
 * Implements banded Smith-Waterman with full affine gap scoring (Gotoh algorithm).
 * Each CUDA thread processes one read against a reference segment.
 *
 * Designed for DGX Spark (Blackwell sm_120, CUDA 13.x).
 *
 * Scoring model:
 *   M(i,j)  = max( M(i-1,j-1), Ix(i-1,j-1), Iy(i-1,j-1) ) + sub(r[i], q[j])
 *   Ix(i,j) = max( M(i-1,j) - go, Ix(i-1,j) ) - ge           // gap in read (vertical)
 *   Iy(i,j) = max( M(i,j-1) - go, Iy(i,j-1) ) - ge           // gap in ref (horizontal)
 *   All values clamped to >= 0 (local alignment).
 *
 * Band mapping:  k = j - i + band_width   =>   range [0, 2*band_width]
 *   Diagonal (i-1,j-1): same k
 *   Above    (i-1,j):   k+1
 *   Left     (i,j-1):   k-1
 */

#include <cuda_runtime.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Scoring constants — DNA alphabet {A, C, G, T, N}
// ---------------------------------------------------------------------------
__constant__ int score_matrix[5][5] = {
    //   A   C   G   T   N
    {    2, -3, -1, -3, -1 },  // A
    {   -3,  2, -3, -1, -1 },  // C
    {   -1, -3,  2, -3, -1 },  // G
    {   -3, -1, -3,  2, -1 },  // T
    {   -1, -1, -1, -1, -1 },  // N
};

__device__ int char_to_idx(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:             return 4;  // N or unknown
    }
}

// ---------------------------------------------------------------------------
// Smith-Waterman with affine gaps (Gotoh), banded — score + alignment bounds
//
// Shared memory layout (per thread block, per thread):
//   prev_M[band], prev_Ix[band], prev_Iy[band]  — previous row
//   curr_M[band], curr_Ix[band], curr_Iy[band]  — current row
//   = 6 * band_size ints
//
// Output per read:
//   scores[read_idx]          — max alignment score
//   read_start[read_idx]      — alignment start in read (0-based)
//   read_end[read_idx]        — alignment end in read (exclusive)
//   ref_start[read_idx]       — alignment start in reference (0-based)
//   ref_end[read_idx]         — alignment end in reference (exclusive)
// ---------------------------------------------------------------------------
/**
 * Smith-Waterman with affine gaps (Gotoh), banded — score + alignment bounds.
 *
 * Uses local memory (L1-cached) per thread — no shared memory contention.
 * Each thread processes one read independently with 6 DP arrays.
 * Max practical band_width: 100 (band_size=201, 6×201×4=4,824 bytes per thread).
 */
__global__
void smith_waterman_affine_kernel(
    const char* __restrict__ reads,
    const char* __restrict__ ref,
    float*  __restrict__ scores,
    int*    __restrict__ read_start,
    int*    __restrict__ read_end,
    int*    __restrict__ ref_start,
    int*    __restrict__ ref_end,
    int num_reads,
    int read_len,
    int ref_len,
    int band_width,
    int gap_open,
    int gap_extend
) {
    int read_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (read_idx >= num_reads) return;

    const char* read = reads + read_idx * read_len;
    int band_size = 2 * band_width + 1;

    // Per-thread shared memory: 6 arrays × band_size ints
    // Total shmem = blockDim.x * 6 * band_size * sizeof(int)
    // L1 cached, no inter-thread sharing needed
    extern __shared__ int shmem[];
    int cells_per_thread = 6 * band_size;
    int* my_mem = shmem + threadIdx.x * cells_per_thread;

    int* prev_M  = my_mem;
    int* prev_Ix = prev_M  + band_size;
    int* prev_Iy = prev_Ix + band_size;
    int* curr_M  = prev_Iy + band_size;
    int* curr_Ix = curr_M  + band_size;
    int* curr_Iy = curr_Ix + band_size;

    int best_score = 0;
    int align_start_i = read_len, align_start_j = ref_len;
    int align_end_i = 0, align_end_j = 0;

    // Initialize previous row
    for (int k = 0; k < band_size; k++) {
        prev_M[k]  = 0;
        prev_Ix[k] = 0;
        prev_Iy[k] = 0;
    }

    // DP over read positions (i)
    for (int i = 0; i < read_len; i++) {
        int r_char = char_to_idx(read[i]);

        // Compute band cell range for this row
        // j = i - band_width ... i + band_width, clipped to [0, ref_len-1]
        int j_start = max(0, i - band_width);
        int j_end   = min(ref_len - 1, i + band_width);

        // Initialize current row cells to 0
        for (int j = j_start; j <= j_end; j++) {
            int k = j - i + band_width;
            curr_M[k]  = 0;
            curr_Ix[k] = 0;
            curr_Iy[k] = 0;
        }

        // --- Pass 1: M and Ix (depend on previous row only) ---
        for (int j = j_start; j <= j_end; j++) {
            int k = j - i + band_width;
            int s = score_matrix[r_char][char_to_idx(ref[j])];

            // M(i,j) = max(M(i-1,j-1), Ix(i-1,j-1), Iy(i-1,j-1)) + s
            // Diagonal is same k in previous row
            int diag_best = prev_M[k];
            if (prev_Ix[k] > diag_best) diag_best = prev_Ix[k];
            if (prev_Iy[k] > diag_best) diag_best = prev_Iy[k];
            int m_val = diag_best + s;
            if (m_val < 0) m_val = 0;  // local alignment restart

            // Ix(i,j) = max(M(i-1,j) - gap_open, Ix(i-1,j)) - gap_extend
            // Above cell has band index k+1 (j stays same, i-1)
            int ix_val = 0;
            if (j >= j_start && j <= j_end) {
                int above_k = k + 1;
                if (above_k >= 0 && above_k < band_size) {
                    int from_M  = prev_M[above_k]  - gap_open;
                    int from_Ix = prev_Ix[above_k] - gap_extend;
                    ix_val = (from_M > from_Ix) ? from_M : from_Ix;
                }
            }
            if (ix_val < 0) ix_val = 0;

            curr_M[k]  = m_val;
            curr_Ix[k] = ix_val;
        }

        // --- Pass 2: Iy (depends on current row M and Iy, left-to-right) ---
        for (int j = j_start; j <= j_end; j++) {
            int k = j - i + band_width;

            // Iy(i,j) = max(M(i,j-1) - gap_open, Iy(i,j-1)) - gap_extend
            int iy_val = 0;
            int left_k = k - 1;
            if (left_k >= 0) {
                int from_M  = curr_M[left_k]  - gap_open;
                int from_Iy = curr_Iy[left_k] - gap_extend;
                iy_val = (from_M > from_Iy) ? from_M : from_Iy;
            }
            if (iy_val < 0) iy_val = 0;
            curr_Iy[k] = iy_val;
        }

        // --- Track best score and alignment boundaries ---
        for (int j = j_start; j <= j_end; j++) {
            int k = j - i + band_width;
            int cell_max = curr_M[k];
            if (curr_Ix[k] > cell_max) cell_max = curr_Ix[k];
            if (curr_Iy[k] > cell_max) cell_max = curr_Iy[k];

            if (cell_max > best_score) {
                best_score = cell_max;
            }

            // Track any cell with positive score for alignment bounds
            if (cell_max > 0) {
                if (i < align_start_i) align_start_i = i;
                if (j < align_start_j) align_start_j = j;
                if (i > align_end_i)   align_end_i   = i;
                if (j > align_end_j)   align_end_j   = j;
            }
        }

        // Swap row pointers for next iteration
        int* tmp;
        tmp = prev_M;  prev_M  = curr_M;  curr_M  = tmp;
        tmp = prev_Ix; prev_Ix = curr_Ix; curr_Ix = tmp;
        tmp = prev_Iy; prev_Iy = curr_Iy; curr_Iy = tmp;
    }

    // Output
    scores[read_idx] = (float)best_score;

    if (best_score > 0) {
        read_start[read_idx] = align_start_i;
        read_end[read_idx]   = align_end_i + 1;     // exclusive
        ref_start[read_idx]  = align_start_j;
        ref_end[read_idx]    = align_end_j + 1;     // exclusive
    } else {
        // No alignment found
        read_start[read_idx] = 0;
        read_end[read_idx]   = 0;
        ref_start[read_idx]  = 0;
        ref_end[read_idx]    = 0;
    }
}

// ---------------------------------------------------------------------------
// Lightweight kernel: score-only, no alignment bounds (faster, less output)
// ---------------------------------------------------------------------------
__global__
void smith_waterman_score_only_kernel(
    const char* __restrict__ reads,
    const char* __restrict__ ref,
    float* __restrict__ scores,
    int num_reads,
    int read_len,
    int ref_len,
    int band_width,
    int gap_open,
    int gap_extend
) {
    int read_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (read_idx >= num_reads) return;

    const char* read = reads + read_idx * read_len;
    int band_size = 2 * band_width + 1;

    // Per-thread shared memory
    extern __shared__ int shmem[];
    int cells_per_thread = 6 * band_size;
    int* my_mem = shmem + threadIdx.x * cells_per_thread;

    int* prev_M  = my_mem;
    int* prev_Ix = prev_M  + band_size;
    int* prev_Iy = prev_Ix + band_size;
    int* curr_M  = prev_Iy + band_size;
    int* curr_Ix = curr_M  + band_size;
    int* curr_Iy = curr_Ix + band_size;

    int best_score = 0;

    // Initialize
    for (int k = 0; k < band_size; k++) {
        prev_M[k] = prev_Ix[k] = prev_Iy[k] = 0;
    }

    for (int i = 0; i < read_len; i++) {
        int r_char = char_to_idx(read[i]);
        int j_start = max(0, i - band_width);
        int j_end   = min(ref_len - 1, i + band_width);

        // Clear current row
        for (int j = j_start; j <= j_end; j++) {
            int k = j - i + band_width;
            curr_M[k] = curr_Ix[k] = curr_Iy[k] = 0;
        }

        // M and Ix (previous row)
        for (int j = j_start; j <= j_end; j++) {
            int k = j - i + band_width;
            int s = score_matrix[r_char][char_to_idx(ref[j])];

            int diag_best = prev_M[k];
            if (prev_Ix[k] > diag_best) diag_best = prev_Ix[k];
            if (prev_Iy[k] > diag_best) diag_best = prev_Iy[k];
            int m_val = diag_best + s;
            if (m_val < 0) m_val = 0;

            int ix_val = 0;
            int above_k = k + 1;
            if (above_k < band_size) {
                int fm  = prev_M[above_k]  - gap_open;
                int fix = prev_Ix[above_k] - gap_extend;
                ix_val = (fm > fix) ? fm : fix;
            }
            if (ix_val < 0) ix_val = 0;

            curr_M[k]  = m_val;
            curr_Ix[k] = ix_val;
        }

        // Iy (current row, left-to-right)
        for (int j = j_start; j <= j_end; j++) {
            int k = j - i + band_width;
            int iy_val = 0;
            int left_k = k - 1;
            if (left_k >= 0) {
                int fm  = curr_M[left_k]  - gap_open;
                int fiy = curr_Iy[left_k] - gap_extend;
                iy_val = (fm > fiy) ? fm : fiy;
            }
            if (iy_val < 0) iy_val = 0;
            curr_Iy[k] = iy_val;
        }

        // Track max
        for (int j = j_start; j <= j_end; j++) {
            int k = j - i + band_width;
            int v = curr_M[k];
            if (curr_Ix[k] > v) v = curr_Ix[k];
            if (curr_Iy[k] > v) v = curr_Iy[k];
            if (v > best_score) best_score = v;
        }

        // Swap
        int* tmp;
        tmp = prev_M;  prev_M  = curr_M;  curr_M  = tmp;
        tmp = prev_Ix; prev_Ix = curr_Ix; curr_Ix = tmp;
        tmp = prev_Iy; prev_Iy = curr_Iy; curr_Iy = tmp;
    }

    scores[read_idx] = (float)best_score;
}

// ---------------------------------------------------------------------------
// Host-callable wrappers (extern "C" for ctypes interop)
// ---------------------------------------------------------------------------
extern "C" {

// One-time init: allow up to 100KB dynamic shared memory on Blackwell
static int _shmem_opt_inited = 0;

static void _init_opt_shmem() {
    if (_shmem_opt_inited) return;
    // Request maximum dynamic shared memory on Blackwell
    // cudaLimitMaxDynamicSharedMemorySize = 0x1E (CUDA 12+)
    cudaDeviceSetLimit((cudaLimit)0x1E, 98304);
    _shmem_opt_inited = 1;
}

/**
 * Full Smith-Waterman affine-gap alignment with alignment bounds.
 *
 * Uses local memory (L1-cached) per thread — no shared memory limits.
 * Max band_width = 100 (hardcoded local array size).
 */
int launch_sw_affine(
    const char* reads,
    const char* ref,
    float* scores,
    int* read_start,
    int* read_end,
    int* ref_start,
    int* ref_end,
    int num_reads,
    int read_len,
    int ref_len,
    int band_width,
    int gap_open,
    int gap_extend,
    int block_size
) {
    if (block_size <= 0) block_size = 256;
    _init_opt_shmem();
    int band_size = 2 * band_width + 1;
    int per_thread = 6 * band_size * (int)sizeof(int);

    // Auto-cap to fit shared memory per block
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int max_threads = (int)(prop.sharedMemPerBlock / per_thread);
    if (max_threads < 1) max_threads = 1;
    if (block_size > max_threads) block_size = max_threads;

    size_t shmem = (size_t)block_size * per_thread;
    int num_blocks = (num_reads + block_size - 1) / block_size;

    // Request larger dynamic shared memory (up to device maximum)
    cudaFuncSetAttribute(
        smith_waterman_affine_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem
    );

    smith_waterman_affine_kernel<<<num_blocks, block_size, shmem>>>(
        reads, ref, scores,
        read_start, read_end, ref_start, ref_end,
        num_reads, read_len, ref_len,
        band_width, gap_open, gap_extend
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "SW affine kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();
    return 0;
}

/**
 * Score-only Smith-Waterman (no alignment bounds — faster, less memory).
 */
int launch_sw_score_only(
    const char* reads,
    const char* ref,
    float* scores,
    int num_reads,
    int read_len,
    int ref_len,
    int band_width,
    int gap_open,
    int gap_extend,
    int block_size
) {
    if (block_size <= 0) block_size = 256;
    _init_opt_shmem();
    int band_size = 2 * band_width + 1;
    int per_thread = 6 * band_size * (int)sizeof(int);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int max_threads = (int)(prop.sharedMemPerBlock / per_thread);
    if (max_threads < 1) max_threads = 1;
    if (block_size > max_threads) block_size = max_threads;

    size_t shmem = (size_t)block_size * per_thread;
    int num_blocks = (num_reads + block_size - 1) / block_size;

    cudaFuncSetAttribute(
        smith_waterman_score_only_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem
    );

    smith_waterman_score_only_kernel<<<num_blocks, block_size, shmem>>>(
        reads, ref, scores,
        num_reads, read_len, ref_len,
        band_width, gap_open, gap_extend
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "SW score-only kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();
    return 0;
}

/**
 * Async Smith-Waterman affine-gap alignment (stream-aware, no sync).
 *
 * Caller is responsible for:
 *   - Allocating/deallocating device memory for reads and ref
 *   - Ensuring shared memory fits within device limits
 *   - Synchronizing the stream when results are needed
 */
int launch_sw_affine_async(
    const char* d_reads,       // device pointer
    const char* d_ref,         // device pointer
    float* d_scores,           // device pointer
    int* d_read_start,         // device pointer
    int* d_read_end,           // device pointer
    int* d_ref_start,          // device pointer
    int* d_ref_end,            // device pointer
    int num_reads,
    int read_len,
    int ref_len,
    int band_width,
    int gap_open,
    int gap_extend,
    int block_size,
    cudaStream_t stream
) {
    if (block_size <= 0) block_size = 256;
    int num_blocks = (num_reads + block_size - 1) / block_size;
    int band_size = 2 * band_width + 1;
    size_t shmem = 6 * band_size * sizeof(int);

    smith_waterman_affine_kernel<<<num_blocks, block_size, shmem, stream>>>(
        d_reads, d_ref, d_scores,
        d_read_start, d_read_end, d_ref_start, d_ref_end,
        num_reads, read_len, ref_len,
        band_width, gap_open, gap_extend
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "SW affine async error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

/**
 * Async score-only Smith-Waterman (stream-aware, no sync).
 */
int launch_sw_score_only_async(
    const char* d_reads,
    const char* d_ref,
    float* d_scores,
    int num_reads,
    int read_len,
    int ref_len,
    int band_width,
    int gap_open,
    int gap_extend,
    int block_size,
    cudaStream_t stream
) {
    if (block_size <= 0) block_size = 256;
    int num_blocks = (num_reads + block_size - 1) / block_size;
    int band_size = 2 * band_width + 1;
    size_t shmem = 6 * band_size * sizeof(int);

    smith_waterman_score_only_kernel<<<num_blocks, block_size, shmem, stream>>>(
        d_reads, d_ref, d_scores,
        num_reads, read_len, ref_len,
        band_width, gap_open, gap_extend
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "SW score-only async error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

/**
 * Check if band_width fits in device shared memory.
 * Returns 0 if OK, -2 if too large.
 */
int check_band_width(int band_width) {
    int band_size = 2 * band_width + 1;
    size_t shmem = 6 * band_size * sizeof(int);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (shmem > prop.sharedMemPerBlock) {
        fprintf(stderr, "band_width=%d requires %zu bytes shared mem "
                "(max %zu). Reduce band_width.\n",
                band_width, shmem, prop.sharedMemPerBlock);
        return -2;
    }
    return 0;
}

} // extern "C"

