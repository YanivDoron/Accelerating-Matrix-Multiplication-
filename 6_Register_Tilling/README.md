## 🧠 Register Tiling for Maximum Throughput

Until now, each thread in the CUDA kernel was responsible for computing a **single output element** per launch. While simple, this method underutilizes the available registers and leads to excessive memory access, where shared or global memory is redundantly read by neighboring threads.

**Register tiling** addresses this inefficiency by assigning each thread to compute a **4×4 tile** of the output matrix, instead of just one element. This strategy allows each thread to **reuse data** already loaded into registers, significantly reducing repeated loads of rows from **matrix A** and columns from **matrix B**.

As block sizes increase (e.g., from 16×16 to 32×32), **shared memory usage grows**, and data reuse becomes even more critical. Register tiling complements shared memory tiling by enabling **reuse within threads**, while shared memory enables reuse **across threads** in a block. This synergy is essential for sustaining **high throughput** and scaling matrix size efficiently.

---

### 🧮 Execution Hierarchy for 1024×1024 GEMM with Register Tiling

| **Level**     | **# of Threads** | **# of Lower-Level Units** | **Total on GPU** | **Computation Per Unit**         |
|---------------|------------------|-----------------------------|------------------|----------------------------------|
| Thread        | 1                | -                           | 65,536           | 4×4 tile of C[i]                 |
| Warp          | 32               | 32                          | 2048             | 32 threads × 4×4 = 512 elements |
| Block         | 64               | 4                           | 1024             | Full 64×64 tile of C             |
| SM (Multiproc)| 1024             | 4                           | 64               | 4 blocks per SM                  |
| Full GPU      | 1024 blocks      | —                           | —                | Computes full 1024×1024 matrix   |

---

> ⚠️ **Disclaimer Regarding Shared Memory Configuration with Register Tiling**  
Adding register tiling to the shared memory implementation allows us to restructure the kernel configuration so that **fewer threads cover larger tile regions**. We reduce the number of threads per block to 16×16, allowing us to **increase the shared memory tile size**. The observed performance gains are thus the result of **three factors**:  
1. Shared memory utilization  
2. Register-level data reuse  
3. Optimized thread/block configuration

---

### 📊 Performance Results

| **Size** | **MKL (ms)** | **Shared Memory (ms)** | **Register Tiling (ms)** | **Spd SM** | **Spd RT** |
|--------:|--------------:|------------------------:|--------------------------:|-----------:|-----------:|
| 128     | 0.17          | 0.10                    | 0.12                      | 1.71×      | 1.40×      |
| 256     | 0.72          | 0.11                    | 0.12                      | 6.23×      | 5.88×      |
| 512     | 4.15          | 0.27                    | 0.15                      | 15.32×     | 27.37×     |
| 1024    | 31.65         | 1.53                    | 0.53                      | 20.71×     | 59.53×     |
| 1408    | 75.35         | 3.81                    | 1.12                      | 19.79×     | 67.26×     |
| 1792    | 153.95        | 6.69                    | 1.80                      | 23.02×     | 85.73×     |
| 2048    | 246.70        | 9.40                    | 2.34                      | 26.24×     | 105.49×    |

---

### 🔍 Improvement Analysis

The results demonstrate a consistent and **significant improvement** in execution time when combining **Shared Memory** with **Register Tiling**. While shared memory alone provides a strong boost over MKL, **Register Tiling achieves even higher speedups** by maximizing register utilization and reducing memory latency.

At small sizes (e.g., 128×128), the difference is minor due to instruction overhead and limited thread utilization.  
However, as matrix size grows, Register Tiling shows **superior scalability**, reaching a **105× speedup** over MKL at 2048×2048—thanks to:
- Fewer memory accesses
- Zero shared memory bank conflicts
- Higher computational throughput per thread


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

Benchmarks the current matrix multiplication implementation across multiple matrix sizes.
compare to CPU-based matrix multiplication using Intel MKL and previeus implementation .
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

## 🧪 `script_benchmark_<X>.sh`
Benchmarks the current matrix multiplication implementation across multiple matrix sizes.
compare to CPU-based matrix multiplication using Intel MKL and previeus implementation .

What it does:

Runs cuurent GPU matrix multiplication across different sizes.

Records runtime and speedups plot to screen.

Output:

**Output:**
- `results/times.npy`: Runtimes (ms) for each method and size.
- `results/speedups.npy`: Speedup compared to baseline (MKL and previeus implementation  ).

```bash
bash script_benchmark_<X>.sh
```
