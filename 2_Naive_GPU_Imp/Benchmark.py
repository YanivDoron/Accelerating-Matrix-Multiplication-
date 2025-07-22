import torch
import time
import numpy as np
from matmul import matmul_naive  # Your custom CUDA kernel
import gc 

# ------------------------------------------
def time_cpu(fn):
    gc.collect()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1e3  # ms

def time_gpu(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    end.synchronize()
    return out, start.elapsed_time(end)  # ms

# ------------------------------------------
def benchmark(size_list, dtype=torch.float32, reps=3):
    print(f"{'Size':>8} | {'MKL (ms)':>10} | {'CUDA (ms)':>10} | {'Speedup':>8}")
    print("-" * 45)

    for N in size_list:
        A_cpu = torch.empty((N, N), dtype=dtype).uniform_(-1, 1)
        B_cpu = torch.empty((N, N), dtype=dtype).uniform_(-1, 1)

        A_gpu = A_cpu.cuda()
        B_gpu = B_cpu.cuda()

        # Warm-up
        _ = A_cpu @ B_cpu
        _ = matmul_naive(A_gpu, B_gpu)

        # MKL
        mkl_times = []
        for _ in range(reps):
            _, t = time_cpu(lambda: A_cpu @ B_cpu)
            mkl_times.append(t)
        mkl_time = np.mean(mkl_times)

        # CUDA naive flat
        cuda_times = []
        for _ in range(reps):
            _, t = time_gpu(lambda: matmul_naive(A_gpu, B_gpu))
            cuda_times.append(t)
        cuda_time = np.mean(cuda_times)

        speedup = mkl_time / cuda_time if cuda_time > 0 else float('inf')

        print(f"{N:8d} | {mkl_time:10.3f} | {cuda_time:10.3f} | {speedup:8.2f}×")

# ------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    sizes = list(range(128, 2048 + 1, 128))  # 128, 256, ..., 2048
    benchmark(sizes)
