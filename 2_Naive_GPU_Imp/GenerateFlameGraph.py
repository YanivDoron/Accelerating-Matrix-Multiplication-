import torch
from torch.profiler import profile, record_function, ProfilerActivity
import matmul_cuda  # Your compiled CUDA extension

TRACE_PATH = "naive_GPU_flamegraph.json"

def matmul_naive(A1, A2):
    return matmul_cuda.matmul_naive(A1.contiguous(), A2.contiguous())

def main():
    N = 2048
    A = torch.randn(N, N, device='cuda')
    B = torch.randn(N, N, device='cuda')

    # Warm-up (important for accurate profiling)
    for _ in range(1):
        matmul_cuda.matmul(A.contiguous(), B.contiguous())


    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) as prof:
        with record_function("matmul_naive"):
            matmul_naive(A, B)

    # Export trace to Chrome JSON format
    prof.export_chrome_trace(TRACE_PATH)
    print(f"✅ Trace saved to: {TRACE_PATH}")

if __name__ == "__main__":
    main()
