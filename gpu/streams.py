"""GPU stream pipeline — multi-stream overlap for DGX Spark (Blackwell).

Implements triple-buffered CUDA stream pipelining to overlap:
  - H2D copy of batch N+1
  - Kernel execution of batch N
  - D2H copy of batch N-1

Uses ctypes to call CUDA runtime (libcudart.so) directly — no PyCUDA required.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from obs.log import log


# ---------------------------------------------------------------------------
# CUDA runtime bindings (libcudart.so)
# ---------------------------------------------------------------------------
class CUDARuntime:
    """Thin ctypes wrapper around libcudart.so for stream + memory ops."""

    def __init__(self):
        lib_path = ctypes.util.find_library("cudart")
        if lib_path is None:
            # Try explicit path for CUDA 13.x on Ubuntu
            lib_path = "/usr/local/cuda/lib64/libcudart.so"
        self._lib = ctypes.CDLL(lib_path)
        self._available = True
        self._setup()

    def _setup(self):
        lib = self._lib

        # cudaStreamCreate
        lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.cudaStreamCreate.restype = ctypes.c_int

        # cudaStreamDestroy
        lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        lib.cudaStreamDestroy.restype = ctypes.c_int

        # cudaStreamSynchronize
        lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        lib.cudaStreamSynchronize.restype = ctypes.c_int

        # cudaMallocHost (pinned memory)
        lib.cudaMallocHost.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
        ]
        lib.cudaMallocHost.restype = ctypes.c_int

        # cudaFreeHost
        lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
        lib.cudaFreeHost.restype = ctypes.c_int

        # cudaMalloc
        lib.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
        ]
        lib.cudaMalloc.restype = ctypes.c_int

        # cudaFree
        lib.cudaFree.argtypes = [ctypes.c_void_p]
        lib.cudaFree.restype = ctypes.c_int

        # cudaMemcpyAsync
        lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_void_p,  # kind, stream
        ]
        lib.cudaMemcpyAsync.restype = ctypes.c_int

        # cudaDeviceSynchronize
        lib.cudaDeviceSynchronize.argtypes = []
        lib.cudaDeviceSynchronize.restype = ctypes.c_int

    def _check(self, ret: int, msg: str = ""):
        if ret != 0:
            raise RuntimeError(f"CUDA error {ret}: {msg}")

    def stream_create(self) -> int:
        s = ctypes.c_void_p()
        self._check(self._lib.cudaStreamCreate(ctypes.byref(s)), "stream_create")
        return s.value

    def stream_destroy(self, stream: int):
        self._lib.cudaStreamDestroy(stream)

    def stream_sync(self, stream: int):
        self._check(self._lib.cudaStreamSynchronize(stream), "stream_sync")

    def device_sync(self):
        self._check(self._lib.cudaDeviceSynchronize(), "device_sync")

    def malloc_host(self, size: int) -> int:
        ptr = ctypes.c_void_p()
        self._check(self._lib.cudaMallocHost(ctypes.byref(ptr), size), "malloc_host")
        return ptr.value

    def free_host(self, ptr: int):
        self._lib.cudaFreeHost(ptr)

    def malloc_device(self, size: int) -> int:
        ptr = ctypes.c_void_p()
        self._check(self._lib.cudaMalloc(ctypes.byref(ptr), size), "malloc_device")
        return ptr.value

    def free_device(self, ptr: int):
        self._lib.cudaFree(ptr)

    def memcpy_htod_async(self, dst: int, src: int, size: int, stream: int):
        # cudaMemcpyHostToDevice = 1
        self._check(
            self._lib.cudaMemcpyAsync(dst, src, size, 1, stream),
            "memcpy_htod_async",
        )

    def memcpy_dtoh_async(self, dst: int, src: int, size: int, stream: int):
        # cudaMemcpyDeviceToHost = 2
        self._check(
            self._lib.cudaMemcpyAsync(dst, src, size, 2, stream),
            "memcpy_dtoh_async",
        )


# ---------------------------------------------------------------------------
# Kernel library (async wrappers)
# ---------------------------------------------------------------------------
class AsyncKernels:
    """Async versions of the SW kernels (stream-aware, device pointers)."""

    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            import os
            lib_path = os.path.join(
                os.path.dirname(__file__), "..", "build", "libcuda_kernels.so",
            )
        self._lib = ctypes.CDLL(lib_path)

        # launch_sw_affine_async
        self._lib.launch_sw_affine_async.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,  # d_reads, d_ref
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # scores, rs, re
            ctypes.c_void_p, ctypes.c_void_p,  # fs, fe
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # n_reads, read_len, ref_len
            ctypes.c_int, ctypes.c_int, ctypes.c_int,  # band_width, gap_open, gap_extend
            ctypes.c_int,                         # block_size
            ctypes.c_void_p,                      # stream
        ]
        self._lib.launch_sw_affine_async.restype = ctypes.c_int

        # launch_sw_score_only_async
        self._lib.launch_sw_score_only_async.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,  # d_reads, d_ref
            ctypes.c_void_p,                    # d_scores
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_void_p,
        ]
        self._lib.launch_sw_score_only_async.restype = ctypes.c_int

        # check_band_width
        self._lib.check_band_width.argtypes = [ctypes.c_int]
        self._lib.check_band_width.restype = ctypes.c_int

    def check_band(self, band_width: int) -> bool:
        return self._lib.check_band_width(band_width) == 0

    def sw_affine_async(
        self,
        d_reads: int, d_ref: int,
        d_scores: int, d_rs: int, d_re: int, d_fs: int, d_fe: int,
        n_reads: int, read_len: int, ref_len: int,
        band_width: int, gap_open: int, gap_extend: int,
        block_size: int, stream: int,
    ) -> int:
        return self._lib.launch_sw_affine_async(
            d_reads, d_ref,
            d_scores, d_rs, d_re, d_fs, d_fe,
            n_reads, read_len, ref_len,
            band_width, gap_open, gap_extend,
            block_size, stream,
        )

    def sw_score_async(
        self,
        d_reads: int, d_ref: int,
        d_scores: int,
        n_reads: int, read_len: int, ref_len: int,
        band_width: int, gap_open: int, gap_extend: int,
        block_size: int, stream: int,
    ) -> int:
        return self._lib.launch_sw_score_only_async(
            d_reads, d_ref, d_scores,
            n_reads, read_len, ref_len,
            band_width, gap_open, gap_extend,
            block_size, stream,
        )


# ---------------------------------------------------------------------------
# Triple-buffer slot
# ---------------------------------------------------------------------------
@dataclass
class StreamSlot:
    """One slot in the triple-buffer pipeline."""
    stream: int                           # CUDA stream handle
    # Pinned host buffers
    h_reads: int                          # pinned host reads buffer
    h_scores: int
    h_read_start: int
    h_read_end: int
    h_ref_start: int
    h_ref_end: int
    # Device buffers
    d_reads: int
    d_ref: int
    d_scores: int
    d_read_start: int
    d_read_end: int
    d_ref_start: int
    d_ref_end: int
    # Sizes
    reads_size: int
    ref_size: int
    n_reads: int = 0
    read_len: int = 0
    ref_len: int = 0
    # Reference pointer (shared d_ref, allocated once)
    ref_owned: bool = False


# ---------------------------------------------------------------------------
# Stream Pipeline
# ---------------------------------------------------------------------------
class StreamPipeline:
    """Triple-buffered CUDA stream pipeline for overlapping copy + compute.

    Pipeline stages (3 slots rotating):
      Slot 0: D2H copy results of batch N-2
      Slot 1: Kernel execution of batch N-1
      Slot 2: H2D copy of batch N

    On Blackwell (sm_120), all 3 operations can execute concurrently.

    Usage:
        pipeline = StreamPipeline(n_reads, read_len, ref_len, band_width, ...)
        pipeline.upload_ref(ref_seq)
        for batch in batches:
            pipeline.submit(batch.reads)
        results = pipeline.collect_all()
        pipeline.cleanup()
    """

    NUM_SLOTS = 3
    BLOCK_SIZE = 256
    # cudaMemcpy kinds
    H2D = 1
    D2H = 2

    def __init__(
        self,
        max_reads_per_batch: int,
        read_len: int,
        ref_len: int,
        band_width: int = 50,
        gap_open: int = 5,
        gap_extend: int = 2,
    ):
        self.max_reads = max_reads_per_batch
        self.read_len = read_len
        self.ref_len = ref_len
        self.band_width = band_width
        self.gap_open = gap_open
        self.gap_extend = gap_extend

        self.cuda = CUDARuntime()
        self.kernels = AsyncKernels()

        # Validate band width
        if not self.kernels.check_band(band_width):
            raise ValueError(
                f"band_width={band_width} exceeds shared memory limit"
            )

        # Buffer sizes (bytes)
        self.reads_buf_size = max_reads_per_batch * read_len  # char
        self.ref_buf_size = ref_len                            # char
        self.scores_buf_size = max_reads_per_batch * 4         # float32
        self.bounds_buf_size = max_reads_per_batch * 4         # int32 each

        # Allocate slots
        self._slots: List[StreamSlot] = []
        for i in range(self.NUM_SLOTS):
            slot = self._allocate_slot()
            self._slots.append(slot)

        # Shared device reference (all slots share one copy)
        self._d_ref_shared = self.cuda.malloc_device(self.ref_buf_size)
        self._ref_uploaded = False

        # Slot tracking
        self._submit_idx = 0       # next slot to submit into
        self._collect_idx = 0      # next slot to collect from
        self._in_flight = 0        # number of submitted uncollected batches
        self._results: List[Optional[Tuple]] = [
            None
        ] * 64  # pre-alloc result slots (ring buffer)
        self._result_cursor = 0

    def _allocate_slot(self) -> StreamSlot:
        s = StreamSlot(
            stream=self.cuda.stream_create(),
            h_reads=self.cuda.malloc_host(self.reads_buf_size),
            h_scores=self.cuda.malloc_host(self.scores_buf_size),
            h_read_start=self.cuda.malloc_host(self.bounds_buf_size),
            h_read_end=self.cuda.malloc_host(self.bounds_buf_size),
            h_ref_start=self.cuda.malloc_host(self.bounds_buf_size),
            h_ref_end=self.cuda.malloc_host(self.bounds_buf_size),
            d_reads=self.cuda.malloc_device(self.reads_buf_size),
            d_ref=0,  # set after upload_ref
            d_scores=self.cuda.malloc_device(self.scores_buf_size),
            d_read_start=self.cuda.malloc_device(self.bounds_buf_size),
            d_read_end=self.cuda.malloc_device(self.bounds_buf_size),
            d_ref_start=self.cuda.malloc_device(self.bounds_buf_size),
            d_ref_end=self.cuda.malloc_device(self.bounds_buf_size),
            reads_size=self.reads_buf_size,
            ref_size=self.ref_buf_size,
        )
        return s

    def upload_ref(self, ref_seq: str):
        """Upload reference sequence to device (once)."""
        ref_bytes = ref_seq[:self.ref_len].ljust(self.ref_len, 'N').encode()
        ref_ptr = ctypes.c_char_p(ref_bytes)

        # Copy to pinned host memory (use slot 0's host buffer temporarily)
        # Actually, use ctypes memmove to pinned host, then H2D
        ctypes.memmove(self._slots[0].h_reads, ref_ptr, self.ref_buf_size)

        # H2D copy on default stream (sync)
        self.cuda.memcpy_htod_async(
            self._d_ref_shared,
            self._slots[0].h_reads,
            self.ref_buf_size,
            0,  # default stream
        )
        self.cuda.device_sync()

        # All slots share the same d_ref
        for slot in self._slots:
            slot.d_ref = self._d_ref_shared
            slot.ref_len = self.ref_len

        self._ref_uploaded = True
        log("stream_ref_uploaded", ref_len=self.ref_len)

    def submit(self, reads: List[str]) -> int:
        """Submit a batch of reads for async processing.

        Returns batch_id for later collection.
        Blocks if pipeline is full (backpressure).
        """
        if not self._ref_uploaded:
            raise RuntimeError("Call upload_ref() before submit()")

        n = len(reads)
        batch_id = self._submit_idx

        # Backpressure: wait for oldest slot if all slots in flight
        if self._in_flight >= self.NUM_SLOTS:
            self._collect_one()

        slot = self._slots[self._submit_idx % self.NUM_SLOTS]

        # Pack reads into pinned host memory
        packed = bytearray(n * self.read_len)
        for i, read in enumerate(reads):
            r = read.ljust(self.read_len, 'N')[:self.read_len]
            packed[i * self.read_len:(i + 1) * self.read_len] = r.encode()

        ctypes.memmove(slot.h_reads, bytes(packed), n * self.read_len)
        slot.n_reads = n
        slot.read_len = self.read_len

        # H2D copy async
        self.cuda.memcpy_htod_async(
            slot.d_reads, slot.h_reads,
            n * self.read_len, slot.stream,
        )

        # Launch kernel on same stream (will wait for H2D to finish)
        ret = self.kernels.sw_affine_async(
            slot.d_reads, slot.d_ref,
            slot.d_scores, slot.d_read_start, slot.d_read_end,
            slot.d_ref_start, slot.d_ref_end,
            n, self.read_len, self.ref_len,
            self.band_width, self.gap_open, self.gap_extend,
            self.BLOCK_SIZE, slot.stream,
        )
        if ret != 0:
            raise RuntimeError(f"Async kernel failed on batch {batch_id}")

        # D2H copy async (will wait for kernel to finish)
        self.cuda.memcpy_dtoh_async(
            slot.h_scores, slot.d_scores,
            n * 4, slot.stream,
        )
        self.cuda.memcpy_dtoh_async(
            slot.h_read_start, slot.d_read_start,
            n * 4, slot.stream,
        )
        self.cuda.memcpy_dtoh_async(
            slot.h_read_end, slot.d_read_end,
            n * 4, slot.stream,
        )
        self.cuda.memcpy_dtoh_async(
            slot.h_ref_start, slot.d_ref_start,
            n * 4, slot.stream,
        )
        self.cuda.memcpy_dtoh_async(
            slot.h_ref_end, slot.d_ref_end,
            n * 4, slot.stream,
        )

        self._submit_idx += 1
        self._in_flight += 1

        return batch_id

    def _collect_one(self) -> Tuple[np.ndarray, ...]:
        """Collect the oldest in-flight batch result."""
        slot = self._slots[self._collect_idx % self.NUM_SLOTS]

        # Synchronize this slot's stream
        self.cuda.stream_sync(slot.stream)

        n = slot.n_reads

        # Read results from pinned host memory into numpy arrays
        scores = np.ctypeslib.as_array(
            ctypes.cast(slot.h_scores, ctypes.POINTER(ctypes.c_float)),
            shape=(n,),
        ).copy()
        rs = np.ctypeslib.as_array(
            ctypes.cast(slot.h_read_start, ctypes.POINTER(ctypes.c_int32)),
            shape=(n,),
        ).copy()
        re = np.ctypeslib.as_array(
            ctypes.cast(slot.h_read_end, ctypes.POINTER(ctypes.c_int32)),
            shape=(n,),
        ).copy()
        fs = np.ctypeslib.as_array(
            ctypes.cast(slot.h_ref_start, ctypes.POINTER(ctypes.c_int32)),
            shape=(n,),
        ).copy()
        fe = np.ctypeslib.as_array(
            ctypes.cast(slot.h_ref_end, ctypes.POINTER(ctypes.c_int32)),
            shape=(n,),
        ).copy()

        self._collect_idx += 1
        self._in_flight -= 1

        return scores, rs, re, fs, fe

    def collect_all(self) -> List[Tuple[np.ndarray, ...]]:
        """Drain all in-flight batches and return results."""
        results = []
        while self._in_flight > 0:
            results.append(self._collect_one())
        return results

    def cleanup(self):
        """Free all CUDA resources."""
        self.cuda.device_sync()

        for slot in self._slots:
            self.cuda.stream_destroy(slot.stream)
            self.cuda.free_host(slot.h_reads)
            self.cuda.free_host(slot.h_scores)
            self.cuda.free_host(slot.h_read_start)
            self.cuda.free_host(slot.h_read_end)
            self.cuda.free_host(slot.h_ref_start)
            self.cuda.free_host(slot.h_ref_end)
            self.cuda.free_device(slot.d_reads)
            self.cuda.free_device(slot.d_scores)
            self.cuda.free_device(slot.d_read_start)
            self.cuda.free_device(slot.d_read_end)
            self.cuda.free_device(slot.d_ref_start)
            self.cuda.free_device(slot.d_ref_end)

        self.cuda.free_device(self._d_ref_shared)
        self._slots.clear()
        log("stream_pipeline_cleanup")


# ---------------------------------------------------------------------------
# Convenience: run a full batch set through the stream pipeline
# ---------------------------------------------------------------------------
def run_stream_pipeline(
    batches: List[List[str]],
    ref_seq: str,
    read_len: int,
    band_width: int = 50,
    gap_open: int = 5,
    gap_extend: int = 2,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Run multiple batches through the triple-buffered stream pipeline.

    Args:
        batches: List of read batches (each is List[str]).
        ref_seq: Reference sequence.
        read_len: Padded read length.
        band_width, gap_open, gap_extend: SW parameters.

    Returns:
        List of (scores, read_start, read_end, ref_start, ref_end) per batch.
    """
    if not batches:
        return []

    max_n = max(len(b) for b in batches)
    ref_len = len(ref_seq)

    log("stream_pipeline_start",
        n_batches=len(batches),
        max_reads_per_batch=max_n,
        ref_len=ref_len,
        band_width=band_width,
    )

    pipeline = StreamPipeline(
        max_reads_per_batch=max_n,
        read_len=read_len,
        ref_len=ref_len,
        band_width=band_width,
        gap_open=gap_open,
        gap_extend=gap_extend,
    )

    try:
        t0 = time.perf_counter()

        pipeline.upload_ref(ref_seq)

        # Submit all batches
        for batch in batches:
            pipeline.submit(batch)

        # Collect all results
        results = pipeline.collect_all()

        elapsed = (time.perf_counter() - t0) * 1000.0
        total_reads = sum(len(b) for b in batches)
        log("stream_pipeline_done",
            n_batches=len(batches),
            n_reads=total_reads,
            time_ms=round(elapsed, 2),
            throughput=round(total_reads / (elapsed / 1000.0), 1),
        )

        return results
    finally:
        pipeline.cleanup()
