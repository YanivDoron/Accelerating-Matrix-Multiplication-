import torch
import time
import numpy as np
import gc
import matmul_cuda

# Define all tested kernels
def matmul(A1, A2):
    return matmul_cuda.matmul(A1.contiguous(), A2.contiguous())

def matmul_block_SM(A1, A2):
    return matmul_cuda.matmul_block_SM(A1.contiguous(), A2.contiguous())

def matmul_streamed_SM(A1, A2):
    return matmul_cuda.matmul_streamed_SM(A1.contiguous(), A2.contiguous())

def matmul_RT_streamed(A1, A2):
    return matmul_cuda.matmul_RT_streamed(A1.contiguous(), A2.contiguous())

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
    print(f"{'Size':>6} | {'MKL (ms)':>10} | {'matmul (ms)':>12} | {'Blk+SM (ms)':>12} | {'Str+SM (ms)':>12} | {'RT+Str (ms)':>12} |"
          f" {'Spd M':>7} | {'Spd BSM':>8} | {'Spd SSM':>8} | {'Spd RT':>8}")
    print("-" * 125)

    for N in size_list:
        A_cpu = torch.empty((N, N), dtype=dtype).uniform_(-1, 1)
        B_cpu = torch.empty((N, N), dtype=dtype).uniform_(-1, 1)

        A_gpu = A_cpu.cuda()
        B_gpu = B_cpu.cuda()

        # Warm-up
        _ = A_cpu @ B_cpu
        _ = matmul(A_gpu, B_gpu)
        _ = matmul_block_SM(A_gpu, B_gpu)
        _ = matmul_streamed_SM(A_gpu, B_gpu)
        _ = matmul_RT_streamed(A_gpu, B_gpu)

        # MKL baseline
        mkl_times = [time_cpu(lambda: A_cpu @ B_cpu)[1] for _ in range(reps)]
        mkl_time = np.mean(mkl_times)

        # matmul
        matmul_time = np.mean([time_gpu(lambda: matmul(A_gpu, B_gpu))[1] for _ in range(reps)])

        # Block + Shared Memory
        blk_sm_time = np.mean([time_gpu(lambda: matmul_block_SM(A_gpu, B_gpu))[1] for _ in range(reps)])

        # Streamed + Shared Memory
        str_sm_time = np.mean([time_gpu(lambda: matmul_streamed_SM(A_gpu, B_gpu))[1] for _ in range(reps)])

        # RT + Streamed
        rt_stream_time = np.mean([time_gpu(lambda: matmul_RT_streamed(A_gpu, B_gpu))[1] for _ in range(reps)])

        # Speedups
        spd_matmul = mkl_time / matmul_time if matmul_time > 0 else float("inf")
        spd_blk_sm = mkl_time / blk_sm_time if blk_sm_time > 0 else float("inf")
        spd_str_sm = mkl_time / str_sm_time if str_sm_time > 0 else float("inf")
        spd_rt = mkl_time / rt_stream_time if rt_stream_time > 0 else float("inf")

        print(f"{N:6d} | {mkl_time:10.3f} | {matmul_time:12.3f} | {blk_sm_time:12.3f} | {str_sm_time:12.3f} | {rt_stream_time:12.3f} |"
              f" {spd_matmul:7.2f}× | {spd_blk_sm:8.2f}× | {spd_str_sm:8.2f}× | {spd_rt:8.2f}×")

# ------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    sizes = list(range(128, 2048 + 1, 128))
    benchmark(sizes)
