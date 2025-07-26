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
// -------------------------------------------------------Shared Memory Block-Wise Matrix Multiplication Kernel On GPU ----------------------------------------
//  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
// A shared memory matrix multiplication kernel that uses a block-wise approach.
// This kernel is designed to use shared memory to improve performance by reducing global memory accesses.
// It divides the workload into blocks, each computing a TILE x TILE submatrix of C.
// The kernel uses shared memory to load tiles of A and B, allowing for faster access during the computation.

constexpr int TILE = 32;
//  ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
// ------------------------------------------------------------
// 1. Shared Memory Block-Tiled Matrix Multiplication (float32)
// ------------------------------------------------------------

__global__ void matmul_kernel_block_shared_fixed(   torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> A,
                                                    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> B,
                                                    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> C,
                                                    int N) {

    // Each block computes a TILE x TILE submatrix of C
    // The block index is calculated based on the block and thread indices
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    // Define shared memory tiles for A and B 
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    

    float sum = 0.0f;
    //Due to limited shared memory size, we will use a loop to load tiles of A and B
    // Each thread load a part of the tile into shared memory from global memory
    // Each thread computes the dot product of the row of A and the column of B
    // The dot product is computed by iterating over the inner dimension N
    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {

        // Calculate the tiled column and row indices
        int tiled_col = t * TILE + threadIdx.x;
        int tiled_row = t * TILE + threadIdx.y;

        //Load tiles of A and B into shared memory
        As[threadIdx.y][threadIdx.x] = (row < N && tiled_col < N) ? A[row][tiled_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (tiled_row < N && col < N) ? B[tiled_row][col] : 0.0f;

        __syncthreads();
        // Compute the dot product for the current tile 
        for (int k = 0; k < TILE; ++k){sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];}
        
        // Synchronize threads to ensure all threads have loaded their tiles before proceeding
        __syncthreads();
    }
    // Store the result in the output matrix C
    if (row < N && col < N){C[row][col] = sum;}
}


// Shared memory block-tiled matrix multiplication kernel Launcher function
at::Tensor matmul_cuda_block_SM(const at::Tensor A, const at::Tensor B) {

    // Check that the input tensors are on CUDA and have the correct dimensions
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda(),   "Tensors must be on CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,                   "Expected 2-D inputs");
    TORCH_CHECK(A.size(1) == B.size(0),                         "Inner dimensions must match"); 
    TORCH_CHECK(A.scalar_type() == torch::kFloat32,             "Only float32 is supported");

    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());


    // Set up grid and block dimensions
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    //Accessors for the input tensors (More readable and efficient access to tensor data)
    auto A_acc = A.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto B_acc = B.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto C_acc = C.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();

    // Launch the kernel with the grid and block dimensions
    matmul_kernel_block_shared_fixed<<<grid, block>>>(A_acc, B_acc, C_acc, static_cast<int>(N));
    cudaDeviceSynchronize();
    return C;
}


// ------------------------------------------------------------------
// 2. Streamed Shared Memory Matrix Multiplication Kernel (float32)
// ------------------------------------------------------------------
__global__ void matmul_kernel_block_shared_tileblock(
    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> A,
    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> B,
    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> C,
    int N, int row_start, int row_end, int col_start, int col_end) {


    // Compute the global row and column this thread is responsible for,
    // shifted by the streamed block offset (row_start, col_start)
    int row = blockIdx.y * TILE + threadIdx.y + row_start;
    int col = blockIdx.x * TILE + threadIdx.x + col_start;

    // Define shared memory tiles for A and B 
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float sum = 0.0f;
    

    // Due to limited shared memory size, we will use a loop to load tiles of A and B
    // Each thread load a part of the tile into shared memory from global memory
    // Each thread computes the dot product of the row of A and the column of B
    // The dot product is computed by iterating over the inner dimension N
    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int tiled_col = t * TILE + threadIdx.x;
        int tiled_row = t * TILE + threadIdx.y;

        //Load tiles of A and B into shared memory
        As[threadIdx.y][threadIdx.x] = (row < row_end && tiled_col < N) ? A[row][tiled_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (tiled_row < N && col < col_end) ? B[tiled_row][col] : 0.0f;

        __syncthreads();
        // Compute the dot product for the current tile 
        for (int k = 0; k < TILE; ++k){sum += As[threadIdx.y][k] * Bs[k][threadIdx.x]; }
            

        __syncthreads();
    }

    if (row < row_end && col < col_end){C[row][col] = sum;}
        
}


at::Tensor matmul_cuda_streamed_SM(const at::Tensor A, const at::Tensor B) {
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Expected 2-D inputs");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "Only float32 is supported");

    
    const int N = A.size(0);
    // Allocate output matrix C on the same device and dtype as A
    auto C = torch::zeros({N, N}, A.options());

    // Determine stream layout: assume NUM_STREAMS is a perfect square
    const int SQRT_STREAMS = static_cast<int>(sqrt(NUM_STREAMS));
    TORCH_CHECK(SQRT_STREAMS * SQRT_STREAMS == NUM_STREAMS, "NUM_STREAMS must be a perfect square");

    int tile_rows = N / SQRT_STREAMS;
    int tile_cols = N / SQRT_STREAMS;

    // Define block shape (number of threads per block)
    dim3 block(TILE, TILE);
    
    // Create NUM_STREAMS CUDA streams for concurrent execution
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i){streams[i] = at::cuda::getStreamFromPool();}
        
    // Get packed accessors to raw tensor data for efficient device accesss
    auto A_acc = A.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto B_acc = B.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto C_acc = C.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();


    // Loop over a grid of tiles: each tile assigned to one CUDA stream
    for (int ty = 0; ty < SQRT_STREAMS; ++ty) {
        for (int tx = 0; tx < SQRT_STREAMS; ++tx) {

            // Define the start and end row/column of the tile this stream is responsible for
            int stream_id = ty * SQRT_STREAMS + tx;

            int row_start = ty * tile_rows;
            int row_end   = (ty == SQRT_STREAMS - 1) ? N : (ty + 1) * tile_rows;

            // Calculate size of the current tile in rows and columns
            int col_start = tx * tile_cols;
            int col_end   = (tx == SQRT_STREAMS - 1) ? N : (tx + 1) * tile_cols;

            int rows = row_end - row_start;
            int cols = col_end - col_start;

            dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

            // Launch kernel for this tile asynchronously on its assigned stream
            matmul_kernel_block_shared_tileblock<<<grid, block, 0, streams[stream_id]>>>(
                A_acc, B_acc, C_acc, N, row_start, row_end, col_start, col_end);
        }
    }

    // Synchronize all streams to ensure all computation is complete
    for (auto& stream : streams){cudaStreamSynchronize(stream); }
        

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");

    return C;
}
