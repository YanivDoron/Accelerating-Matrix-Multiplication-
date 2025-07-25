# 🧠 Optimized CPU Matrix Multiplication (Intel MKL Reference)
### Previous  : [GPU Architecture ](/0_GPU_Arch)                           Next  : [Block Wise Tilling](/3_Block_Wise_Tilling) 

## 📌 Overview

This module serves as the **CPU-side reference baseline** for performance comparisons in our matrix multiplication project.  
It uses **Intel’s Math Kernel Library (MKL)** to perform high-performance dense matrix multiplication.

Intel MKL is a highly optimized, multithreaded library specifically engineered for numerical computations on modern CPUs.  
By leveraging MKL, we can approach peak CPU performance **without** manually writing low-level SIMD or multithreaded code.

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
![Performance Plot](images/gragh1.png)
![Performance Plot](images/profiling.png)

---
# 📊 Running and Profiling the Matrix Multiplication Implementation

This section explains how to run, profile, and visualize the performance of the different matrix multiplication implementations included in this project. Each script below plays a specific role in benchmarking GPU and CPU performance, generating visual output, and saving logs for analysis.

---

## 🔧 Scripts Overview

| Script                   | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `Benchmark.py`           | Benchmarks the current method and show their runtime + speedup data         |
| `GenerateFlameGraph.py`  | Profiles a single implementation and generates a Chrome trace timeline      |
| `ShowPerformance.py`     | Profiles a single implementation and store the data on text file            |
| `script_benchmark_<X>.sh`| Benchmarks the current matmul against the previeus implementation           |                     

---

## 📊 `Benchmark.py`

Benchmarks the selected matrix multiplication implementations across multiple matrix sizes.

**What it does:**
- Runs current implementation
- Measures runtime and calculates speedups.

**Output:**
- `results/times.npy`: Runtimes (ms) for each method and size.
- `results/speedups.npy`: Speedup compared to baseline (e.g., MKL or naive).

**To run:**
```bash
python Benchmark.py
```
---
## 🔥 `GenerateFlameGraph.py`

Profiles a single kernel execution using torch.profiler and generates a Chrome-compatible trace.

Output:

log/profile.json: Timeline that can be loaded into chrome://tracing.

```bash
python GenerateFlameGraph.py
```

## 📄 `ShowPerformance.py`

Profiles a single kernel execution using torch.profiler and store the data on text file.

Output:
Profile.txt

```bash
python ShowPerformance.py
```

## 🧪 `script_benchmark_MKL.sh`
Benchmarks CPU-based matrix multiplication using Intel MKL (e.g., via NumPy or SciPy).

What it does:

Runs CPU matrix multiplication across different sizes.

Records runtime into .npy files.

Output:

**Output:**
- `results/times.npy`: Runtimes (ms) for each method and size.
- `results/speedups.npy`: Speedup compared to baseline (MKL and previeus implementation  ).

```bash
bash script_benchmark_MKL.sh
```





