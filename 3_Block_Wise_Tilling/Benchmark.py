import torch
import time
import numpy as np
import gc
from matmul import matmul_naive, matmul_naive_block  # ✅ Make sure both kernels are available

# ------------------------------------------
def time_cpu(fn):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
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
    print(f"{'Size':>6} | {'MKL (ms)':>10} | {'Naive (ms)':>11} | {'Blk (ms)':>10} | {'Spd N':>7} | {'Spd B':>7}")
    print("-" * 60)

    for N in size_list:
        A_cpu = torch.empty((N, N), dtype=dtype).uniform_(-1, 1)
        B_cpu = torch.empty((N, N), dtype=dtype).uniform_(-1, 1)

        A_gpu = A_cpu.cuda()
        B_gpu = B_cpu.cuda()

        # Warm-up
        _ = A_cpu @ B_cpu
        _ = matmul_naive(A_gpu, B_gpu)
        _ = matmul_naive_block(A_gpu, B_gpu)

        # MKL
        mkl_times = [time_cpu(lambda: A_cpu @ B_cpu)[1] for _ in range(reps)]
        mkl_time = np.mean(mkl_times)

        # Naive CUDA
        naive_times = [time_gpu(lambda: matmul_naive(A_gpu, B_gpu))[1] for _ in range(reps)]
        naive_time = np.mean(naive_times)

        # Blocked CUDA
        block_times = [time_gpu(lambda: matmul_naive_block(A_gpu, B_gpu))[1] for _ in range(reps)]
        block_time = np.mean(block_times)

        spd_naive = mkl_time / naive_time if naive_time > 0 else float("inf")
        spd_block = mkl_time / block_time if block_time > 0 else float("inf")

        print(f"{N:6d} | {mkl_time:10.3f} | {naive_time:11.3f} | {block_time:10.3f} | {spd_naive:7.2f}× | {spd_block:7.2f}×")

# ------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    sizes = list(range(128, 2048 + 1, 128))
    benchmark(sizes)
