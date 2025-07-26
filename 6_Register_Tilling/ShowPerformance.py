import torch
import torch.profiler
import os
import matmul_cuda  # ✅ Ensure this module has matmul

def matmul_tested(A1, A2):
    return matmul_cuda.matmul(A1.contiguous(), A2.contiguous())

# --------------------------------------------------------------------------
def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = 1024  # Matrix size
    A = torch.randn(N, N, device=device)
    B = torch.randn(N, N, device=device)

    kernel_name = matmul_tested.__code__.co_names[0]
    logdir = os.path.join("./log/profile", kernel_name)
    os.makedirs(logdir, exist_ok=True)

    print(f"\n🔍 Profiling kernel: {kernel_name}")
    print(f"📦 Matrix size: {N}x{N}")
    print(f"💻 Device: {device}")

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(logdir),
        record_shapes=True,
        with_stack=False,
        with_flops=True,
        profile_memory=True,
    ) as prof:
        for _ in range(5):
            C = matmul_tested(A, B)
            prof.step()

    # Save summary
    summary_txt = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
    output_file = f"{kernel_name}_profile.txt"
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, "w") as f:
        f.write(f"Profiling Summary for {kernel_name}\n")
        f.write(summary_txt)

    print(f"\n✅ Profile summary saved to: {output_file}")
    print(f"📁 Trace saved to: {logdir}")
    print(f"📈 View in TensorBoard:\n    tensorboard --logdir={logdir}")

if __name__ == "__main__":
    main()
