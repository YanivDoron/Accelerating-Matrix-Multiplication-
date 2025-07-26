import torch
import time
import numpy as np
import gc
from matmul import (
    matmul_naive_block,
    matmul_streamed_tileblock  # ✅ Make sure it's defined in your extension
)

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
    print(f"{'Size':>6} | {'MKL (ms)':>10} | {'Blk (ms)':>10} | {'Stream (ms)':>12} | {'Spd B':>7} | {'Spd S':>7}")
    print("-" * 70)

    for N in size_list:
        A_cpu = torch.empty((N, N), dtype=dtype).uniform_(-1, 1)
        B_cpu = torch.empty((N, N), dtype=dtype).uniform_(-1, 1)

        A_gpu = A_cpu.cuda()
        B_gpu = B_cpu.cuda()

        # Warm-up
        _ = A_cpu @ B_cpu
        _ = matmul_naive_block(A_gpu, B_gpu)
        _ = matmul_streamed_tileblock(A_gpu, B_gpu)

        # MKL
        mkl_times = [time_cpu(lambda: A_cpu @ B_cpu)[1] for _ in range(reps)]
        mkl_time = np.mean(mkl_times)

        # Blocked CUDA
        block_times = [time_gpu(lambda: matmul_naive_block(A_gpu, B_gpu))[1] for _ in range(reps)]
        block_time = np.mean(block_times)

        # Streamed CUDA
        stream_times = [time_gpu(lambda: matmul_streamed_tileblock(A_gpu, B_gpu))[1] for _ in range(reps)]
        stream_time = np.mean(stream_times)

        spd_block = mkl_time / block_time if block_time > 0 else float("inf")
        spd_stream = mkl_time / stream_time if stream_time > 0 else float("inf")

        print(f"{N:6d} | {mkl_time:10.3f} | {block_time:10.3f} | {stream_time:12.3f} | {spd_block:7.2f}× | {spd_stream:7.2f}×")

# ------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    sizes = list(range(128, 2048 + 1, 128))  # 128, 256, ..., 2048
    benchmark(sizes)
