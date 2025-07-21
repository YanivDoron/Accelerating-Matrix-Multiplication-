![Performance Plot](images/LOGOS.png)
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

- ***Optimized Implementation on CPU (as a Reference Point)*** [Here](1_MKL)

  Establishes a performance baseline using CPU-optimized techniques such as loop unrolling, cache-friendly access patterns, and Intel MKL. This helps quantify speedup from GPU versions.

- ***Naive GPU Implementation (Significant Performance Improvement)*** [Here](2_Naive_GPU_Imp)  

  A simple CUDA kernel that assigns one thread per output element, using global memory directly. While it benefits from massive parallelism, it suffers from poor memory access and no data reuse.

- ***Block-Wise Tiling for GPU Parallelism (Slight Performance Improvement)***  [Here](3_Block_Wise_Tilling)
  
  Divides the matrix into tiles and maps them to thread blocks. This improves coalesced memory access and reduces redundant global memory reads, allowing better GPU occupancy.

- ***Streaming (No Significant Change in Performance – used mainly for debug and analysis)***  [Here](4_Streaming)
       
  Introduces multiple CUDA streams to overlap data transfers and compute with kernel execution, helping hide memory latency and improve throughput.

- ***Leveraging Shared Memory for Data Reuse (Significant Performance Improvement)*** [Here](5_Shared_Memory)
   
  Copies tiles of input matrices into fast shared memory, allowing threads to reuse data multiple times and drastically reduce global memory traffic. This improves bandwidth efficiency.

- ***Register Tiling for Maximum Throughput (Significant Performance Improvement)***  [Here](6_Register_Tilling)
  
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

## Install

**(1) Recommended:** Install mamba as in: [https://github.com/conda-forge/miniforge#mambaforge](https://github.com/conda-forge/miniforge#mambaforge).  
If you prefer to use conda, proceed to (2) and replace "mamba" commands with "conda" commands.

**(1.1) First remove any previous Anaconda/miniconda installation via:**
```sh
conda activate base
conda/mamba install anaconda-clean
anaconda-clean --yes
rm -rf ~/miniconda3
rm -rf ~/miniforge3
rm -rf ~/anaconda3
rm -rf ~/.anaconda_backup
```

**(1.2) Install mamba as in:** [https://github.com/conda-forge/miniforge#mambaforge](https://github.com/conda-forge/miniforge#mambaforge)
```sh
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh
conda config --set auto_activate_base false
```

**(2) Install MMUL_proj environment:**
```sh
mamba create -n MMUL_proj
mamba activate MMUL_proj
mamba install python==3.10.2 pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 libgcc-ng==9.5.0 ninja==1.10.2 setuptools==69.5.1 mkl=2024.0.0 cuda-compiler=11.7 cuda-version=11.7 cuda-nvprune=11.7 cuda-cuxxfilt=11.7 cuda-nvcc=11.7 cuda-cuobjdump=11.7 cuda-cudart-dev=11.7 cuda-cccl=11.7 cuda-nvrtc-dev=11.7 cuda-libraries-dev=11.7 -c pytorch -c nvidia
```