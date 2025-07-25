# ⚡ GPU Matrix Multiplication Benchmark & Profiling Suite

This project contains multiple optimized **CUDA implementations** of square matrix multiplication, ranging from a naive kernel to advanced tiled versions.  
It provides benchmark scripts, profiler tools, and performance visualizations to evaluate and compare GPU acceleration versus CPU baselines (e.g., Intel MKL).

---

## 🚀 Implementations

---

## ⚡ Naive CUDA Implementation

To establish a baseline for GPU acceleration benefits, we implemented a **naive CUDA kernel** that performs square matrix multiplication without using shared memory or register tiling.  
Each CUDA thread is assigned to compute a single output element in the result matrix.

Despite its simplicity, this naive kernel **consistently outperforms Intel MKL**, achieving up to **10.8× speedup** on large matrices (e.g., 1024×1024), thanks to the GPU’s inherent architectural advantages.

---

### ✅ Architectural Advantages of Naive CUDA:

- **Massive Parallelism**  
  Thousands of threads execute in parallel, far beyond what even multicore CPUs with SIMD can handle.

- **High Memory Bandwidth**  
  Even without shared memory, the kernel benefits from fast global memory access.

- **Latency Hiding via Warp Scheduling**  
  Warp-level switching keeps execution going even if one warp stalls.

- **Graceful Scaling**  
  GPU occupancy increases with larger matrices, leading to better performance scaling.

---

### 🧪 Benchmark Results – Naive CUDA

| **Size** | **MKL (ms)** | **Naive (ms)** | **Speedup** |
|----------|--------------|----------------|-------------|
| 128×128  | 0.123        | 0.102          | 1.20×       |
| 256×256  | 0.534        | 0.099          | 5.40×       |
| 512×512  | 3.900        | 0.330          | 11.83×      |
| 1024×1024| 26.826       | 2.944          | 9.11×       |

![Performance Plot – Naive CUDA](images/graph_naive.png)

---

### 🔥 Flame Graph Analysis – Naive CUDA

The execution starts from `flamenaive.py`, calls `matmul_naive()` → PyCapsule → CUDA kernel `matmul_kernel_naive_flat`.  
Due to PyTorch's async model, CUDA work is deferred and only executed upon sync (`cudaDeviceSynchronize`).

![Flame Graph – Naive](images/flame_naive.png)

---

## 🧱 Block-Wise Tiling for GPU Parallelism

To address naive limitations, we implemented a **block-wise tiled CUDA kernel**, assigning a 16×16 thread block to each matrix tile.  
Each thread computes one output element — enabling **scalable**, **high-occupancy** GPU utilization.

---

### ⚠️ Limitation of the Naive Kernel

- Uses 1 block of 1024 threads → engages only 1 SM.
- Low GPU occupancy.
- Not scalable to large matrices.

---

### ✅ Improvements from Block-Wise Tiling

- 🧠 **More Blocks → More SMs Active**
- 🎯 **Better Warp Utilization (256 threads per block)**
- 📶 **Better Memory Coalescing**
- 🔁 **Scalable Across Matrix Sizes**

---

### 📊 Benchmark Results – Blocked CUDA

| **Size** | **MKL (ms)** | **Naive (ms)** | **Block (ms)** | **Naive Speedup** | **Block Speedup** |
|----------|--------------|----------------|----------------|-------------------|-------------------|
| 128×128  | 0.123        | 0.102          | 0.077          | 1.20×             | 1.59×             |
| 256×256  | 0.534        | 0.099          | 0.084          | 5.40×             | 6.37×             |
| 512×512  | 3.900        | 0.330          | 0.349          | 11.83×            | 11.18×            |
| 1024×1024| 26.826       | 2.944          | 2.354          | 9.11×             | 11.40×            |
| 2048×2048| 201.054      | 18.137         | 15.945         | 11.09×            | 12.61×            |

![Performance Plot – Blocked](images/graph_block.png)

---

### 🔥 Flame Graph – Block-Wise CUDA

Each block computes a matrix tile → multiple blocks across SMs = full device utilization.  
Memory latency is better hidden due to concurrency.

![Flame Graph – Block](images/flame_block.png)

---

## 📌 Summary

| Feature                  | Naive CUDA                | Block-Wise CUDA            |
|--------------------------|---------------------------|----------------------------|
| Threads per block        | 1024                      | 256 (16×16)                |
| Number of blocks         | 1                         | (N / 16) × (N / 16)        |
| SM Utilization           | ❌ Low                    | ✅ High                    |
| Scalability              | ❌ Poor                   | ✅ Excellent               |
| Speedup vs MKL (max)     | ~11×                      | ~12.6×                     |

---

## 📊 Running and Profiling the Matrix Multiplication Implementations

This section explains how to benchmark, profile, and analyze all GPU implementations in the project.

---

### 🔧 Scripts Overview

| Script                   | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `Benchmark.py`           | Benchmarks all implementations and prints speedup vs MKL                   |
| `GenerateFlameGraph.py`  | Runs profiler, saves Chrome timeline (log/profile.json)                    |
| `ShowPerformance.py`     | Runs profiler and saves tabular data to `Profile.txt`                      |
| `script_benchmark_<X>.sh`| Bash runner for specific kernels across matrix sizes                        |

---

### 📈 `Benchmark.py`

**What it does:**
- Runs all GPU matmul implementations on several matrix sizes.
- Compares performance to MKL and previous kernels.

**Output:**
- `results/times.npy`
- `results/speedups.npy`

**Usage:**
```bash
python Benchmark.py
