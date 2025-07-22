import torch
import time
import numpy as np

def time_cpu(fn):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1e3  # milliseconds

def benchmark_mkl_matmul(sizes, dtype=torch.float32, reps=3):
    device = torch.device('cpu')
    torch.set_num_threads(torch.get_num_threads())  # ensure max threading (MKL/OpenBLAS)

    print(f"{'Size':>8} | {'Time (ms)':>10} | {'GFLOPs':>10}")
    print("-" * 34)

    for N in sizes:
        A = torch.empty((N, N), dtype=dtype, device=device).uniform_(-1, 1)
        B = torch.empty((N, N), dtype=dtype, device=device).uniform_(-1, 1)

        # Warm-up
        _ = A @ B

        # Timing over multiple runs
        times = []
        for _ in range(reps):
            _, t_ms = time_cpu(lambda: A @ B)
            times.append(t_ms)

        avg_time = np.mean(times)
        gflops = (2 * N ** 3) / (avg_time * 1e6)

        print(f"{N:>8} | {avg_time:10.3f} | {gflops:10.2f}")

if __name__ == "__main__":
    sizes = list(range(128, 2048 + 1, 128))  # 128, 256, ..., 2048
    benchmark_mkl_matmul(sizes)
