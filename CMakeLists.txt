cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# CUDA toolkit — MUST be set BEFORE project() call
# ---------------------------------------------------------------------------
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
# Workaround for aarch64 math-vector.h errors during compiler ID
set(CMAKE_CUDA_FLAGS "-D__CUDA_NO_HALF_OPERATORS__ --expt-relaxed-constexpr" CACHE STRING "" FORCE)

project(hyb_align LANGUAGES CXX CUDA)

# CUDA standard and architecture
# DGX Spark GB10 = Blackwell = sm_120 (compute capability 12.1)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA toolkit (modern CMake)
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# ---------------------------------------------------------------------------
# Build CUDA kernel library
# ---------------------------------------------------------------------------
add_library(cuda_kernels SHARED
    cuda/align_kernel.cu
    cuda/seed_kernel.cu
)

set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(cuda_kernels PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(cuda_kernels
    CUDA::cudart
    CUDA::cuda_driver
)

# Install rules
install(TARGETS cuda_kernels
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
