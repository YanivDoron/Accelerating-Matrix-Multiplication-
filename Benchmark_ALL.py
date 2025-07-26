import torch
import time
import numpy as np
import gc
import matmul_cuda

# ------------------------------------------
# Define all kernels to test
def matmul(A1, A2):
    return matmul_cuda.matmul(A1.contiguous(), A2.contiguous())

def matmul_naive(A1, A2):
    return matmul_cuda.matmul_naive(A1.contiguous(), A2.contiguous())

def matmul_naive_block(A1, A2):
    return matmul_cuda.matmul_naive_block(A1.contiguous(), A2.contiguous())

def matmul_block_SM(A1, A2):
    return matmul_cuda.matmul_block_SM(A1.contiguous(), A2.contiguous())

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
    print(f"{'Size':>6} | {'MKL (ms)':>10} | {'matmul':>9} | {'naive':>9} | {'blk':>9} | {'blk+SM':>9} | {'RTstr':>9} |"
          f" {'spd_m':>7} | {'spd_n':>7} | {'spd_b':>7} | {'spd_bsm':>9} | {'spd_rt':>8}")
    print("-" * 125)

    for N in size_list:
        A_cpu = torch.empty((N, N), dtype=dtype).uniform_(-1, 1)
        B_cpu = torch.empty((N, N), dtype=dtype).uniform_(-1, 1)

        A_gpu = A_cpu.cuda()
        B_gpu = B_cpu.cuda()

        # Warm-up
        _ = A_cpu @ B_cpu
        _ = matmul(A_gpu, B_gpu)
        _ = matmul_naive(A_gpu, B_gpu)
        _ = matmul_naive_block(A_gpu, B_gpu)
        _ = matmul_block_SM(A_gpu, B_gpu)
        _ = matmul_RT_streamed(A_gpu, B_gpu)

        # MKL baseline
        mkl_time = np.mean([time_cpu(lambda: A_cpu @ B_cpu)[1] for _ in range(reps)])

        # CUDA kernels
        time_matmul     = np.mean([time_gpu(lambda: matmul(A_gpu, B_gpu))[1] for _ in range(reps)])
        time_naive      = np.mean([time_gpu(lambda: matmul_naive(A_gpu, B_gpu))[1] for _ in range(reps)])
        time_blk        = np.mean([time_gpu(lambda: matmul_naive_block(A_gpu, B_gpu))[1] for _ in range(reps)])
        time_blk_sm     = np.mean([time_gpu(lambda: matmul_block_SM(A_gpu, B_gpu))[1] for _ in range(reps)])
        time_rt_stream  = np.mean([time_gpu(lambda: matmul_RT_streamed(A_gpu, B_gpu))[1] for _ in range(reps)])

        # Speedups
        spd_matmul = mkl_time / time_matmul if time_matmul > 0 else float("inf")
        spd_naive  = mkl_time / time_naive if time_naive > 0 else float("inf")
        spd_blk    = mkl_time / time_blk if time_blk > 0 else float("inf")
        spd_bsm    = mkl_time / time_blk_sm if time_blk_sm > 0 else float("inf")
        spd_rt     = mkl_time / time_rt_stream if time_rt_stream > 0 else float("inf")

        print(f"{N:6d} | {mkl_time:10.3f} | {time_matmul:9.3f} | {time_naive:9.3f} | {time_blk:9.3f} |"
              f" {time_blk_sm:9.3f} | {time_rt_stream:9.3f} | {spd_matmul:7.2f}× | {spd_naive:7.2f}× |"
              f" {spd_blk:7.2f}× | {spd_bsm:9.2f}× | {spd_rt:8.2f}×")

# ------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    sizes = list(range(128, 2048 + 1, 128))
    benchmark(sizes)
