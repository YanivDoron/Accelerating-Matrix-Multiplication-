# 🧠 Optimized CPU Matrix Multiplication (Intel MKL Reference)

## 📌 Overview

This module serves as the **CPU-side reference baseline** for performance comparisons in our matrix multiplication project.  
It uses **Intel’s Math Kernel Library (MKL)** to perform high-performance dense matrix multiplication.

Intel MKL is a highly optimized, multithreaded library specifically engineered for numerical computations on modern CPUs.  
By leveraging MKL, we can approach peak CPU performance **without** manually writing low-level SIMD or multithreaded code.
![Performance Plot](images/gragh1.png)
---

## ⚙️ Why Intel MKL?

Intel MKL provides architecture-aware optimizations across multiple dimensions:

### ✅ Key Features:

- **Multithreading**  
  MKL automatically parallelizes matrix multiplication across all available CPU cores using efficient internal threading mechanisms.

- **SIMD Vectorization**  
  Uses modern vector instruction sets (e.g., AVX2, AVX-512) to compute multiple elements in parallel within each core.

- **Cache-Aware Blocking**  
  Implements intelligent tiling strategies that optimize data reuse across L1/L2/L3 cache levels, reducing memory access latency.

- **NUMA Awareness**  
  On multi-socket systems, MKL minimizes memory access latency by scheduling threads and memory allocations according to NUMA topology.

- **Highly Tuned Assembly Kernels**  
  Intel engineers provide hand-optimized assembly routines that outperform compiler-generated code, yielding near-hardware-limit performance.

---

## 🧪 Usage as Benchmark

In our project, the MKL implementation is used to:

- Establish a **CPU performance baseline** for matrix multiplication.
- Compare and quantify the **performance gains** achieved by our GPU-based CUDA kernels.

---

## 🚀 How to Use

Make sure Intel MKL is installed and linked correctly (via Anaconda, pip wheels with MKL, or system-level installation):

```bash
# Python example using NumPy with MKL backend (default on many systems)
import numpy as np

A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.matmul(A, B)  # Under the hood, uses MKL
