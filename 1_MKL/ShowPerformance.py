import torch
import torch.profiler
import os

# 🔧 Use torch.matmul which uses MKL on CPU
def matmul_mkl(A1, A2):
    return torch.matmul(A1, A2)

# ------------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    device = torch.device("cpu")  # Force CPU to ensure MKL usage

    N = 1024  # Matrix size
    A = torch.randn(N, N, device=device)
    B = torch.randn(N, N, device=device)

    kernel_name = "matmul_mkl"

    # 🗂 Absolute path to the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 📁 TensorBoard logs go into ./log/profile/<kernel_name>
    tb_logdir = os.path.join(script_dir, "log", "profile", kernel_name)
    os.makedirs(tb_logdir, exist_ok=True)

    # 📄 Save Profile.txt in the same folder as this script
    profile_txt_path = os.path.join(script_dir, "Profile.txt")

    print(f"\n🔍 Profiling kernel: {kernel_name}")
    print(f"📦 Matrix size: {N}x{N}")
    print(f"💻 Device: {device}")

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        activities=[torch.profiler.ProfilerActivity.CPU],
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_logdir),
        record_shapes=True,
        with_stack=False,
        with_flops=True,
        profile_memory=True,
    ) as prof:
        for _ in range(5):
            C = matmul_mkl(A, B)
            prof.step()

    cpu_summary = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)

    print("\n📊 CPU time summary:")
    print(cpu_summary)

    # 💾 Save summary to Profile.txt in script directory
    with open(profile_txt_path, "w") as f:
        f.write(f"Kernel: {kernel_name}\n")
        f.write(f"Matrix size: {N}x{N}\n")
        f.write(f"Device: {device}\n\n")
        f.write("=== CPU Time Summary ===\n")
        f.write(cpu_summary + "\n")

    print(f"\n✅ Summary saved to: {profile_txt_path}")
    print(f"📈 View trace in TensorBoard: tensorboard --logdir={tb_logdir}")

if __name__ == "__main__":
    main()
