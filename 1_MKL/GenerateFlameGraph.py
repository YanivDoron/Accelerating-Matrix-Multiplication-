import torch
from torch.profiler import profile, record_function, ProfilerActivity

TRACE_PATH = "MKL_FLAME.json"

def matmul_mkl(A1, A2):
    return torch.matmul(A1, A2)

def main():
    N = 2048
    A = torch.randn(N, N, device='cpu')  # CPU tensors
    B = torch.randn(N, N, device='cpu')

    # Warm-up (important for accurate profiling)
    for _ in range(1):
        torch.matmul(A, B)

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) as prof:
        with record_function("matmul_mkl"):
            matmul_mkl(A, B)

    prof.export_chrome_trace(TRACE_PATH)
    print(f"✅ MKL CPU trace saved to: {TRACE_PATH}")

if __name__ == "__main__":
    main()
