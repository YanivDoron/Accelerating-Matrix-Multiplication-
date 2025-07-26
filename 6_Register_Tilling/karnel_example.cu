#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>  // for getStreamFromPool and stream utils
#include <vector>



#define TILE_M 64
#define TILE_N 64
#define TILE_K 32
#define WPT 4          // Work per thread (register tiling: 4x4)
#define THREADS_X (TILE_N / WPT)
#define THREADS_Y (TILE_M / WPT) 
constexpr int NUM_STREAMS = 4;  // Number of CUDA streams to use



// ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
// --------------------------------------------------------Final CUDA Kernel for Matrix Multiplication---------------------------------------------------------
// ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
// Fully unrolled 4x4 register tiling with static shared-memory tiled GEMM
// use the following optimization techniques:
// 1. Fully unrolled 4x4 register tiling
// 2. Dynamic shared memory tiling with larger shared memory tile
// 3. Avoid array indexing overhead by using register tiling
// 4. Optimize Block-wise tiling update due to the fact that each thread computes a 4x4 submatrix of C
// ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

__global__ void matmul_kernel(  torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> A,
                                torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> B,
                                torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> C,
                                int N) {


    /// // Calculate block and thread indices
    // Each block computes a TILE_M x TILE_N submatrix of C
    int block_row = blockIdx.y;         // Block row index
    int block_col = blockIdx.x;         // Block column index
    int tx = threadIdx.x;               // Thread index in the x dimension in the block
    int ty = threadIdx.y;               // Thread index in the y dimension in the block
    int tid = ty * blockDim.x + tx;     // Global thread index in the block
    
    int row_base = block_row * TILE_M + ty * WPT;
    int col_base = block_col * TILE_N + tx * WPT;

    // Register tile for accumulating results
    // Each thread computes a 4x4 submatrix of C 
    // Each thread holds 4x4 one float register tile 
    // this is a fully unrolled 4x4 register tiling and avoid array indexing overhead
    float r00=0, r01=0, r02=0, r03=0;
    float r10=0, r11=0, r12=0, r13=0;
    float r20=0, r21=0, r22=0, r23=0;
    float r30=0, r31=0, r32=0, r33=0;


    //Shared memory tiles for A and B
    // Each tile is TILE_M x TILE_K for A and TILE_K x TILE_N for
    // B, where TILE_M and TILE_N are the dimensions of the submatrix C
    // that this block computes, and TILE_K is the inner dimension.    
    __shared__ float Asub[TILE_M][TILE_K];
    __shared__ float Bsub[TILE_K][TILE_N];

    int threads_per_block = blockDim.x * blockDim.y;

    //Load tiles dynamically
    // Each block iterates over the K dimension in chunks of TILE_K in order to load the tiles
    for (int kk = 0; kk < N; kk += TILE_K) {
        // total elements to load
        int A_elems = TILE_M * TILE_K;
        int B_elems = TILE_K * TILE_N;

        // Load A tile dynamically to shared memory
        for (int i = tid; i < A_elems; i += threads_per_block) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int global_row = block_row * TILE_M + row;
            int global_col = kk + col;
            Asub[row][col] = (global_row < N && global_col < N) ? A[global_row][global_col] : 0.0f;
        }

        // Load B tile dynamically to shared memory
        for (int i = tid; i < B_elems; i += threads_per_block) {
            int row = i / TILE_N;
            int col = i % TILE_N;
            int global_row = kk + row;
            int global_col = block_col * TILE_N + col;
            Bsub[row][col] = (global_row < N && global_col < N) ? B[global_row][global_col] : 0.0f;
        }

        __syncthreads();

        // Compute the 4x4 submatrix of C
        // Each thread computes a 4x4 submatrix of C using the loaded tiles
        // Fully unrolled 4x4 register tiling
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

            // Accumulate results in the register tile
            // Each thread computes a 4x4 submatrix of C
            r00 += a0 * b0; r01 += a0 * b1; r02 += a0 * b2; r03 += a0 * b3;
            r10 += a1 * b0; r11 += a1 * b1; r12 += a1 * b2; r13 += a1 * b3;
            r20 += a2 * b0; r21 += a2 * b1; r22 += a2 * b2; r23 += a2 * b3;
            r30 += a3 * b0; r31 += a3 * b1; r32 += a3 * b2; r33 += a3 * b3;
        }

        __syncthreads();
    }
 //
    // Store result to global memory
    // Each thread writes its computed 4x4 submatrix of C to global memory
    #pragma unroll
    for (int i = 0; i < WPT; ++i) {
        for (int j = 0; j < WPT; ++j) {
            int row = row_base + i;
            int col = col_base + j;
            // Use register tiling to avoid array indexing overhead
            // Each thread writes its computed value to the global memory
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

     
// Register tiling with dynamic shared memory Luncher function
at::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Expected 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Dimension mismatch");

    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options().dtype(torch::kFloat32));

    // Set up grid and block dimensions
    // the grid is divided into blocks of size TILE_M x TILE_N
    // using the register tiling technique we can compute a 4x4 submatrix of C and maximize the occupancy
    dim3 blockSize(THREADS_X, THREADS_Y);
    dim3 gridSize((N + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M);

    matmul_kernel<<<gridSize, blockSize>>>(
        A.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        B.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        C.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>(),
        N);

    return C;
}



__global__ void matmul_kernel_REGTILE_streamed( torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> A,
                                                torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> B,
                                                torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> C,
                                                int N, int row_start, int row_end,  int col_start, int col_end) {



    /// // Calculate block and thread indices
    // Each block computes a TILE_M x TILE_N submatrix of C
    int block_row = blockIdx.y;         // Block row index
    int block_col = blockIdx.x;         // Block column index
    int tx = threadIdx.x;               // Thread index in the x dimension in the block
    int ty = threadIdx.y;               // Thread index in the y dimension in the block
    int tid = ty * blockDim.x + tx;     // Global thread index in the block

    //Stream offset
    int row_base = row_start + block_row * TILE_M + ty * WPT;  //  offset row
    int col_base = col_start + block_col * TILE_N + tx * WPT;  //  offset col

    // Register tile for accumulating results
    // Each thread computes a 4x4 submatrix of C 
    // Each thread holds 4x4 one float register tile 
    // this is a fully unrolled 4x4 register tiling and avoid array indexing overhead
    float r00=0, r01=0, r02=0, r03=0;
    float r10=0, r11=0, r12=0, r13=0;
    float r20=0, r21=0, r22=0, r23=0;
    float r30=0, r31=0, r32=0, r33=0;

    //Shared memory tiles for A and B
    // Each tile is TILE_M x TILE_K for A and TILE_K x TILE_N for
    // B, where TILE_M and TILE_N are the dimensions of the submatrix C
    // that this block computes, and TILE_K is the inner dimension.   
    __shared__ float Asub[TILE_M][TILE_K];
    __shared__ float Bsub[TILE_K][TILE_N];

    int threads_per_block = blockDim.x * blockDim.y;


    //Load tiles dynamically
    // Each block iterates over the K dimension in chunks of TILE_K in order to load the tiles
    for (int kk = 0; kk < N; kk += TILE_K) {
        int A_elems = TILE_M * TILE_K;
        int B_elems = TILE_K * TILE_N;

        // Load A tile dynamically to shared memory
        for (int i = tid; i < A_elems; i += threads_per_block) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int global_row = row_start + block_row * TILE_M + row;
            int global_col = kk + col;
            Asub[row][col] = (global_row < row_end && global_col < N) ? A[global_row][global_col] : 0.0f;
        }
        // Load B tile dynamically to shared memory
        for (int i = tid; i < B_elems; i += threads_per_block) {
            int row = i / TILE_N;
            int col = i % TILE_N;
            int global_row = kk + row;
            int global_col = col_start + block_col * TILE_N + col;
            Bsub[row][col] = (global_row < N && global_col < col_end) ? B[global_row][global_col] : 0.0f;
        }

        __syncthreads();

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

    // כתיבה לזיכרון גלובלי
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

            if (row < row_end && col < col_end)
                C[row][col] = val;
        }
    }
}


at::Tensor matmul_RT_streamed_cuda(torch::Tensor A, torch::Tensor B) {
        TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Expected 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Dimension mismatch");

    const int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options().dtype(torch::kFloat32));

    const int SQRT_STREAMS = static_cast<int>(sqrt(NUM_STREAMS));
    TORCH_CHECK(SQRT_STREAMS * SQRT_STREAMS == NUM_STREAMS, "NUM_STREAMS must be a perfect square");

    int tile_rows = (N + SQRT_STREAMS - 1) / SQRT_STREAMS;
    int tile_cols = (N + SQRT_STREAMS - 1) / SQRT_STREAMS;

    dim3 blockSize(THREADS_X, THREADS_Y);

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i)
        streams[i] = at::cuda::getStreamFromPool();

    auto A_acc = A.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto B_acc = B.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto C_acc = C.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();

    for (int ty = 0; ty < SQRT_STREAMS; ++ty) {
        for (int tx = 0; tx < SQRT_STREAMS; ++tx) {
            int stream_id = ty * SQRT_STREAMS + tx;

            int row_start = ty * tile_rows;
            int row_end   = std::min(N, (ty + 1) * tile_rows);

            int col_start = tx * tile_cols;
            int col_end   = std::min(N, (tx + 1) * tile_cols);

            int rows = row_end - row_start;
            int cols = col_end - col_start;

            dim3 gridSize((cols + TILE_N - 1) / TILE_N, (rows + TILE_M - 1) / TILE_M);

            matmul_kernel_REGTILE_streamed<<<gridSize, blockSize, 0, streams[stream_id]>>>(
                A_acc, B_acc, C_acc, N,
                row_start, row_end,
                col_start, col_end
            );
        }
    }

    for (auto& stream : streams)
        cudaStreamSynchronize(stream);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "TileBlock kernel launch failed");

    return C;
}