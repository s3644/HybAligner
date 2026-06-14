"""HybAligner: Hybrid CPU-GPU sequence aligner targeting DGX Spark.

A CUDA-accelerated sequence alignment pipeline combining CPU-based
chaining with GPU-based banded alignment kernels.
"""

from setuptools import setup, find_packages
import subprocess
import os

# Build CUDA kernels via CMake before installing Python package
def build_cuda_extension():
    """Build CUDA shared library via CMake."""
    build_dir = os.path.join(os.path.dirname(__file__), "build")
    os.makedirs(build_dir, exist_ok=True)

    subprocess.check_call(["cmake", ".."], cwd=build_dir)
    subprocess.check_call(["make", f"-j{os.cpu_count()}"], cwd=build_dir)

    print(f"CUDA kernels built in {build_dir}")

setup(
    name="hyb_align",
    version="0.5.0",
    description="Hybrid CPU-GPU sequence aligner for DGX Spark",
    author="HybAligner Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "psutil>=5.9",
        "tqdm>=4.65",
    ],
    extras_require={
        "gpu": [
            "cupy-cuda12x>=13.0",
            "pycuda>=2024.1",
        ],
        "dev": [
            "pytest>=7.4",
            "pytest-benchmark>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hyb-align=hyb_align:main",
        ],
    },
)
