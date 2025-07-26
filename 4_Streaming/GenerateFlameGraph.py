import torch
from torch.profiler import profile, record_function, ProfilerActivity
import matmul  # Your compiled CUDA extension module

TRACE_PATH = "stream_GPU_flamegraph.json"

def matmul_stream(A1, A2):
    return matmul.matmul_streamed_tileblock(A1.contiguous(), A2.contiguous())

def main():
    N = 2048
    A = torch.randn(N, N, device='cuda')
    B = torch.randn(N, N, device='cuda')

    # Warm-up (important for accurate profiling)
    for _ in range(1):
        matmul_stream(A, B)

    # Start profiling
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) as prof:
        with record_function("matmul_stream"):
            matmul_stream(A, B)

    # Export trace to Chrome trace format
    prof.export_chrome_trace(TRACE_PATH)
    print(f"✅ Trace saved to: {TRACE_PATH}")

if __name__ == "__main__":
    main()
