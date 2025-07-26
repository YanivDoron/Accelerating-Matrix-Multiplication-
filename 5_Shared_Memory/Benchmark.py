import torch
import time
import numpy as np
import gc
import matmul_cuda  # Make sure this module contains all the following functions

# Define each tested kernel
def matmul_naive_block(A1, A2):
    return matmul_cuda.matmul_naive_block(A1.contiguous(), A2.contiguous())

def matmul_streamed(A1, A2):
    return matmul_cuda.matmul_streamed(A1.contiguous(), A2.contiguous())

def matmul_block_SM(A1, A2):
    return matmul_cuda.matmul_block_SM(A1.contiguous(), A2.contiguous())

def matmul_streamed_SM(A1, A2):
    return matmul_cuda.matmul_streamed_SM(A1.contiguous(), A2.contiguous())

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
    print(f"{'Size':>6} | {'MKL (ms)':>10} | {'Blk (ms)':>10} | {'Stream (ms)':>10} | {'Blk+SM (ms)':>12} | {'Str+SM (ms)':>12} | {'Spd B':>7} | {'Spd S':>7} | {'Spd BSM':>8} | {'Spd SSM':>8}")
    print("-" * 110)

    for N in size_list:
        A_cpu = torch.empty((N, N), dtype=dtype).uniform_(-1, 1)
        B_cpu = torch.empty((N, N), dtype=dtype).uniform_(-1, 1)

        A_gpu = A_cpu.cuda()
        B_gpu = B_cpu.cuda()

        # Warm-up
        _ = A_cpu @ B_cpu
        _ = matmul_naive_block(A_gpu, B_gpu)
        _ = matmul_streamed(A_gpu, B_gpu)
        _ = matmul_block_SM(A_gpu, B_gpu)
        _ = matmul_streamed_SM(A_gpu, B_gpu)

        # MKL
        mkl_times = [time_cpu(lambda: A_cpu @ B_cpu)[1] for _ in range(reps)]
        mkl_time = np.mean(mkl_times)

        blk_times = [time_gpu(lambda: matmul_naive_block(A_gpu, B_gpu))[1] for _ in range(reps)]
        blk_time = np.mean(blk_times)

        str_times = [time_gpu(lambda: matmul_streamed(A_gpu, B_gpu))[1] for _ in range(reps)]
        str_time = np.mean(str_times)

        blk_sm_times = [time_gpu(lambda: matmul_block_SM(A_gpu, B_gpu))[1] for _ in range(reps)]
        blk_sm_time = np.mean(blk_sm_times)

        str_sm_times = [time_gpu(lambda: matmul_streamed_SM(A_gpu, B_gpu))[1] for _ in range(reps)]
        str_sm_time = np.mean(str_sm_times)

        print(f"{N:6d} | {mkl_time:10.3f} | {blk_time:10.3f} | {str_time:10.3f} | {blk_sm_time:12.3f} | {str_sm_time:12.3f} |"
              f" {mkl_time/blk_time:7.2f}× | {mkl_time/str_time:7.2f}× | {mkl_time/blk_sm_time:8.2f}× | {mkl_time/str_sm_time:8.2f}×")

# ------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    sizes = list(range(128, 2048 + 1, 128))
    benchmark(sizes)
