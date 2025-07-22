#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_M 64
#define TILE_N 64
#define TILE_K 32
#define WPT 4          // Work per thread (register tiling: 4x4)
#define THREADS_X (TILE_N / WPT)
#define THREADS_Y (TILE_M / WPT) //////


// Fully unrolled 4x4 register tiling with static shared-memory tiled GEMM
__global__ void matmul_kernel(
    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> A,
    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> B,
    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> C,
    int N) {

    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int row_base = block_row * TILE_M + ty * WPT;
    int col_base = block_col * TILE_N + tx * WPT;

    // Register tile
    float r00=0, r01=0, r02=0, r03=0;
    float r10=0, r11=0, r12=0, r13=0;
    float r20=0, r21=0, r22=0, r23=0;
    float r30=0, r31=0, r32=0, r33=0;

    __shared__ float Asub[TILE_M][TILE_K];
    __shared__ float Bsub[TILE_K][TILE_N];

    int threads_per_block = blockDim.x * blockDim.y;

    for (int kk = 0; kk < N; kk += TILE_K) {
        // total elements to load
        int A_elems = TILE_M * TILE_K;
        int B_elems = TILE_K * TILE_N;

        // Load A tile dynamically
        for (int i = tid; i < A_elems; i += threads_per_block) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int global_row = block_row * TILE_M + row;
            int global_col = kk + col;
            Asub[row][col] = (global_row < N && global_col < N) ? A[global_row][global_col] : 0.0f;
        }

        // Load B tile dynamically
        for (int i = tid; i < B_elems; i += threads_per_block) {
            int row = i / TILE_N;
            int col = i % TILE_N;
            int global_row = kk + row;
            int global_col = block_col * TILE_N + col;
            Bsub[row][col] = (global_row < N && global_col < N) ? B[global_row][global_col] : 0.0f;
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float a0 = Asub[ty * WPT + 0][k];
            float a1 = Asub[ty * WPT + 1][k];
            float a2 = Asub[ty * WPT + 2][k];
            float a3 = Asub[ty * WPT + 3][k];

            float b0 = Bsub[k][tx * WPT + 0];
            float b1 = Bsub[k][tx * WPT + 1];
            float b2 = Bsub[k][tx * WPT + 2];
            float b3 = Bsub[k][tx * WPT + 3];

            r00 += a0 * b0; r01 += a0 * b1; r02 += a0 * b2; r03 += a0 * b3;
            r10 += a1 * b0; r11 += a1 * b1; r12 += a1 * b2; r13 += a1 * b3;
            r20 += a2 * b0; r21 += a2 * b1; r22 += a2 * b2; r23 += a2 * b3;
            r30 += a3 * b0; r31 += a3 * b1; r32 += a3 * b2; r33 += a3 * b3;
        }

        __syncthreads();
    }

    // Store result
    #pragma unroll
    for (int i = 0; i < WPT; ++i) {
        for (int j = 0; j < WPT; ++j) {
            int row = row_base + i;
            int col = col_base + j;

            float val =
                (i == 0 && j == 0) ? r00 : (i == 0 && j == 1) ? r01 : (i == 0 && j == 2) ? r02 : (i == 0 && j == 3) ? r03 :
                (i == 1 && j == 0) ? r10 : (i == 1 && j == 1) ? r11 : (i == 1 && j == 2) ? r12 : (i == 1 && j == 3) ? r13 :
                (i == 2 && j == 0) ? r20 : (i == 2 && j == 1) ? r21 : (i == 2 && j == 2) ? r22 : (i == 2 && j == 3) ? r23 :
                (i == 3 && j == 0) ? r30 : (i == 3 && j == 1) ? r31 : (i == 3 && j == 2) ? r32 : r33;

            if (row < N && col < N)
                C[row][col] = val;
        }
    }
}

at::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Expected 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Dimension mismatch");

    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options().dtype(torch::kFloat32));

    dim3 blockSize(THREADS_X, THREADS_Y);
    dim3 gridSize((N + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M);

    matmul_kernel<<<gridSize, blockSize>>>(
        A.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        B.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        C.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        N);

    return C;
}


at::Tensor matmul_RT_streamed_cuda(torch::Tensor A, torch::Tensor B) {

}


// ────────────────────────────────────────────────────────────────────────────────
// matmul_naive.cu
// A *very* simple (O(N³) global-memory) GEMM for square matrices.
// ────────────────────────────────────────────────────────────────────────────────




template <typename scalar_t>
__global__ void matmul_kernel_naive_flat(
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> A,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> B,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> C,
    const int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;

    int row = idx / N;
    int col = idx % N;

    scalar_t sum = 0;
    for (int k = 0; k < N; ++k) {
        sum += A[row][k] * B[k][col];
    }
    C[row][col] = sum;
}


at::Tensor matmul_cuda_naive(const at::Tensor A, const at::Tensor B)
{
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda(),
                "Tensors must be on CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Expected 2-D inputs");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    const int64_t N = A.size(0);                // square matrix assumption
    auto C = torch::zeros({N, N}, A.options());

    const int threads = 1024;
    const int total = N * N;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        A.scalar_type(), "matmul_cuda_naive_flat_launch", ([&] {
            matmul_kernel_naive_flat<scalar_t><<<blocks, threads>>>(
                A.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                B.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                C.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                static_cast<int>(N));
        }));

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "matmul_kernel_naive_flat launch failed");

    return C;
}




template <typename scalar_t>
__global__ void matmul_kernel_naive_block(
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> A,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> B,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> C,
        const int N)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[row][k] * B[k][col];
        }
        C[row][col] = sum;
    }
}

// ────────────────────────────────────────────────────────────────────────────────
// C++/ATen front-end
// ────────────────────────────────────────────────────────────────────────────────
at::Tensor matmul_cuda_naive_block(const at::Tensor A, const at::Tensor B) 
{
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda(),
                "Tensors must be on CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Expected 2-D inputs");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    const int64_t N = A.size(0);               // square assumption
    auto C = torch::zeros({N, N}, A.options());

    // <<< grid, block >>> configuration
    constexpr int TILE = 16;                   // 16×16 threads per block
    const dim3 block(TILE, TILE);
    const dim3 grid((N + TILE - 1) / TILE,
                    (N + TILE - 1) / TILE);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        A.scalar_type(), "matmul_cuda_naive_launch", ([&] {
            matmul_kernel_naive_block<scalar_t><<<grid, block>>>(
                A.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                B.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                C.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
                static_cast<int>(N));
        }));

    // good practice: catch kernel‐launch errors in debug builds
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
                "matmul_kernel_naive launch failed");

    return C;
}



#include <ATen/cuda/CUDAContext.h>  // for getStreamFromPool and stream utils


constexpr int NUM_STREAMS = 4;  // Number of CUDA streams to use

template <typename scalar_t>
__global__ void matmul_kernel_naive_block_streamed(
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> A,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> B,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> C,
    int N,
    int row_start,
    int row_end) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y + row_start;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < row_end && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[row][k] * B[k][col];
        }
        C[row][col] = sum;
    }
}


at::Tensor matmul_cuda_streamed(const at::Tensor A, const at::Tensor B) 
{
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda(),
                "Tensors must be on CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Expected 2-D inputs");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    const int64_t N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    constexpr int TILE = 16;                   // 16×16 threads per block

    const dim3 block(TILE, TILE);
    const dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE / NUM_STREAMS);

    // Divide workload
    int rows_per_stream = (N + NUM_STREAMS - 1) / NUM_STREAMS;

    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i)
        streams[i] = at::cuda::getStreamFromPool();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        A.scalar_type(), "matmul_cuda_naive_block_streamed", ([&] {
            auto A_acc = A.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>();
            auto B_acc = B.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>();
            auto C_acc = C.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>();

            for (int s = 0; s < NUM_STREAMS; ++s) {
                int row_start = s * rows_per_stream;
                int row_end   = std::min((s + 1) * rows_per_stream, (int)N);

                int rows_this = row_end - row_start;
                dim3 stream_grid((N + TILE - 1) / TILE, (rows_this + TILE - 1) / TILE);

                matmul_kernel_naive_block_streamed<scalar_t><<<stream_grid, block, 0, streams[s]>>>(
                    A_acc, B_acc, C_acc, N, row_start, row_end);
            }
        }));

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamSynchronize(streams[i]);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");

    return C;
}

// =============================================
// 1. BLOCK-WISE SHARED MEMORY IMPLEMENTATION
// =============================================

constexpr int TILE = 32;

template <typename scalar_t>
__global__ void matmul_kernel_block_shared_fixed(
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> A,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> B,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> C,
    int N) {

    __shared__ scalar_t As[TILE][TILE];
    __shared__ scalar_t Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    scalar_t sum = 0;

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int tiled_col = t * TILE + threadIdx.x;
        int tiled_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < N && tiled_col < N) ? A[row][tiled_col] : static_cast<scalar_t>(0);
        Bs[threadIdx.y][threadIdx.x] = (tiled_row < N && col < N) ? B[tiled_row][col] : static_cast<scalar_t>(0);

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row][col] = sum;
}

at::Tensor matmul_cuda_block_SM(const at::Tensor A, const at::Tensor B) {
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Expected 2-D inputs");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "matmul_cuda_block_SM", [&] {
        matmul_kernel_block_shared_fixed<scalar_t><<<grid, block>>>(
            A.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            B.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            C.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            N);
    });

    cudaDeviceSynchronize();
    return C;
}

// =============================================
// 2. STREAMED BLOCK-WISE SHARED MEMORY VERSION
// =============================================

template <typename scalar_t>
__global__ void matmul_kernel_block_shared_streamed_fixed(
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> A,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> B,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> C,
    int N,
    int row_offset,
    int row_limit) {

    __shared__ scalar_t As[TILE][TILE];
    __shared__ scalar_t Bs[TILE][TILE];

    int local_row = blockIdx.y * TILE + threadIdx.y;
    int row = local_row + row_offset;
    int col = blockIdx.x * TILE + threadIdx.x;

    scalar_t sum = 0;

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int tiled_col = t * TILE + threadIdx.x;
        int tiled_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < row_limit && tiled_col < N) ? A[row][tiled_col] : static_cast<scalar_t>(0);
        Bs[threadIdx.y][threadIdx.x] = (tiled_row < N && col < N) ? B[tiled_row][col] : static_cast<scalar_t>(0);

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < row_limit && col < N)
        C[row][col] = sum;
}

at::Tensor matmul_cuda_streamed_SM(const at::Tensor A, const at::Tensor B) {
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Expected 2-D inputs");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");

    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 block(TILE, TILE);
    int rows_per_stream = (N + NUM_STREAMS - 1) / NUM_STREAMS;

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i)
        streams[i] = at::cuda::getStreamFromPool();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "matmul_cuda_streamed_SM", [&] {
        auto A_acc = A.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>();
        auto B_acc = B.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>();
        auto C_acc = C.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>();

        for (int s = 0; s < NUM_STREAMS; ++s) {
            int row_start = s * rows_per_stream;
            int row_end = std::min((s + 1) * rows_per_stream, N);
            int rows_this = row_end - row_start;

            dim3 grid((N + TILE - 1) / TILE, (rows_this + TILE - 1) / TILE);

            matmul_kernel_block_shared_streamed_fixed<scalar_t><<<grid, block, 0, streams[s]>>>(
                A_acc, B_acc, C_acc, N, row_start, row_end);
        }
    });

    for (auto& stream : streams)
        cudaStreamSynchronize(stream);

    return C;
}
