# Accelerating Matrix Multiplication on Nvidia GPU

## 📌 Introduction

This project focuses on optimizing matrix multiplication performance on NVIDIA GPUs using CUDA.  
Matrix multiplication is a fundamental operation in scientific computing and machine learning workloads.  
Optimizing it can significantly improve overall system performance.

To accelerate the computation, we developed a custom CUDA kernel optimized for large matrix sizes.  
The kernel utilizes:

- **Block-Wise Tiling for GPU Parallelism**  
- **Streaming Method**  
- **Leveraging Shared Memory for Data Reuse**
- **Register Tiling for Maximum Throughput**

These techniques reduce global memory traffic and improve data reuse and throughput.

## ⚙️ Problem Definition

Given two input matrices  
**A, B ∈ ℝⁿˣⁿ**, with values in **[-1, 1]**,  
compute their matrix product: C = A × B

Where each element in the result matrix C is computed as: C[i][j] = ∑ A[i][k] * B[k][j]
he implementation aims to compute this efficiently on the GPU using CUDA.

## 📚 Chronological Progression of GPU Optimizations

This project explores the step-by-step enhancement of matrix multiplication performance through GPU acceleration. We begin by setting a strong CPU baseline and progressively implement and evaluate GPU optimizations. Each step focuses on exposing more parallelism, reducing memory latency, and increasing arithmetic throughput. At every stage, we analyze what improvements were effective and what challenges were encountered, gradually building toward a highly optimized GPU kernel.

- **Optimized Implementation on CPU (as a Reference Point)**
  
  Establishes a performance baseline using CPU-optimized techniques such as loop unrolling, cache-friendly access patterns, and Intel MKL. This helps quantify speedup from GPU versions.

- **Naive GPU Implementation (Significant Performance Improvement)**  
  A simple CUDA kernel that assigns one thread per output element, using global memory directly. While it benefits from massive parallelism, it suffers from poor memory access and no data reuse.

- **Block-Wise Tiling for GPU Parallelism (Slight Performance Improvement)**  
  Divides the matrix into tiles and maps them to thread blocks. This improves coalesced memory access and reduces redundant global memory reads, allowing better GPU occupancy.

- **Streaming (No Significant Change in Performance – used mainly for debug and analysis)**  
  Introduces multiple CUDA streams to overlap data transfers and compute with kernel execution, helping hide memory latency and improve throughput.

- **Leveraging Shared Memory for Data Reuse (Significant Performance Improvement)**  
  Copies tiles of input matrices into fast shared memory, allowing threads to reuse data multiple times and drastically reduce global memory traffic. This improves bandwidth efficiency.

- **Register Tiling for Maximum Throughput (Significant Performance Improvement)**  
  Each thread computes a small output sub-block using registers. This minimizes memory access and leverages low-latency register files for peak arithmetic throughput.

## 📈 Benchmark Results

Compared to Intel MKL and Our Castum CUDA kernel:

- Achieved **~50× speedup over MKL** on 2048×2048 FP32 matrices.
- Significant improvements over naive CUDA by optimizing memory access patterns and thread workloads.

## ✅ Simplifying Assumptions

- **Contiguous Memory:** All matrices are stored contiguously (row-major).
- **Device-Resident:** All matrices reside in GPU memory. No host-device transfers during computation.
- **FP32 Precision:** All computations are done in 32-bit float (no quantization).
- **No Quantization:** This implementation avoids any form of reduced precision for maximum accuracy.

## 🖥️ Requirements

- CUDA-enabled NVIDIA GPU (e.g., RTX 2080 Ti)
- Python + PyTorch (with CUDA support)
- C++/CUDA build environment (`nvcc`, `setuptools`)


