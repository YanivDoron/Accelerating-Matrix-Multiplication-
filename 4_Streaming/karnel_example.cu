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
// -------------------------------------------------------Streamed Block-Wise Matrix Multiplication Kernel On GPU ---------------------------------------------
//  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
// A streamed matrix multiplication kernel that uses a block-wise approach.
// This kernel is designed to work with multiple CUDA streams to overlap computation and memory transfers.
// It divides the workload into multiple streams, each processing a subset of rows.

// Each block computes a TILE x TILE submatrix of C.
// This kernel is a simple implementation that introduces a block-wise tiling strategy with streaming.

// Fixed block size of 16x16 threads will improve performance over the flat kernel.

// note: This kernel is designed to work with multiple CUDA streams to overlap computation and memory transfers.
//       the method doesnt gain much performance over the naive block-wise kernel, but it is a good example of how to use CUDA streams.
//────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

__global__ void matmul_kernel_naive_block_streamed( torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> A,
                                                    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> B,
                                                    torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> C,
                                                    int N, int row_start, int row_end)  {

    // Each block computes a TILE x TILE submatrix of C
    // The block index is calculated based on the block and thread indices
    // with the row_start and row_end parameters to limit the rows processed by each stream
    int row = blockIdx.y * blockDim.y + threadIdx.y + row_start;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row > row_end || col > N) {return; }

    // Each thread computes the dot product of the row of A and the column of B
    // The dot product is computed by iterating over the inner dimension N
    float sum = 0;
    for (int k = 0; k < N; ++k) { sum += A[row][k] * B[k][col];}
    
    // Store the result in the output matrix C
    C[row][col] = sum;
}


at::Tensor matmul_cuda_streamed_row(const at::Tensor A, const at::Tensor B) {

    // Check that the input tensors are on CUDA and have the correct dimensions
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda(),   "Tensors must be on CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,                   "Expected 2-D inputs");
    TORCH_CHECK(A.size(1) == B.size(0),                         "Inner dimensions must match"); 

    const int64_t N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());
    constexpr int TILE = 16;                   // 16×16 threads per block


    // Set up grid and block dimensions , note that we use a fixed block size of 16x16 threads
    // This block size is a good compromise between occupancy and performance
    // The grid is divided into blocks of size TILE x TILE with multiple streams dividing the workload
    const dim3 block(TILE, TILE);
    const dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE / NUM_STREAMS);


    // Divide workload into multiple streams 
    // Each stream will handle a subset of rows in the output matrix C
    // This allows for overlapping computation and memory transfers
    int rows_per_stream = (N + NUM_STREAMS - 1) / NUM_STREAMS;



    //Start the CUDA streams addition part 
    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);


    // Initialize streams
    for (int i = 0; i < NUM_STREAMS; ++i){
        streams[i] = at::cuda::getStreamFromPool();
    }
        
    //Accessors for the input tensors (More readable and efficient access to tensor data)
    auto A_acc = A.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto B_acc = B.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto C_acc = C.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    
    
    // Launch the kernel for each stream
    // Each stream will handle a subset of rows in the output matrix C
    for (int s = 0; s < NUM_STREAMS; ++s) {

        // Calculate the start and end rows for this stream
        // Each stream will handle a subset of rows in the output matrix C
        // each block inside the stream will compute a TILE x TILE submatrix of C
        int row_start = s * rows_per_stream;
        int row_end   = std::min((s + 1) * rows_per_stream, (int)N);
        int rows_this = row_end - row_start;

        // Set up the grid for this stream 
        dim3 stream_grid((N + TILE - 1) / TILE, (rows_this + TILE - 1) / TILE);
        
        // Launch the kernel for this stream of the subset of rows 
        matmul_kernel_naive_block_streamed<<<stream_grid, block, 0, streams[s]>>>( A_acc, B_acc, C_acc, N, row_start, row_end);
    }


    // Synchronize all streams completion
    // This ensures that all streams have completed their work before returning the result
    for (int i = 0; i < NUM_STREAMS; ++i) {cudaStreamSynchronize(streams[i]); }

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");

    return C;
}


__global__ void matmul_kernel_naive_block_streamed(  torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> A,
                                                     torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> B,
                                                     torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> C,
                                                    int N, int row_start, int row_end, int col_start, int col_end) {

    // Compute the global row and column this thread is responsible for,
    // shifted by the streamed block offset (row_start, col_start)
    int row = blockIdx.y * blockDim.y + threadIdx.y + row_start;
    int col = blockIdx.x * blockDim.x + threadIdx.x + col_start;

    // Skip computation if the thread is outside the assigned streaming tile
    if (row >= row_end || col >= col_end) return;

    // Compute the dot product for C[row][col]
    float sum = 0.0f;
    for (int k = 0; k < N; ++k) { sum += A[row][k] * B[k][col]; }

    // Store the result in the output matrix
    C[row][col] = sum;
}


at::Tensor matmul_cuda_streamed_tiled_blocks(const at::Tensor A, const at::Tensor B) {
    // Check input validity: both tensors must be on CUDA and 2D
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,                  "Expected 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0),                        "Inner dimensions must match");
    
    
    constexpr int TILE = 16;   
    const int64_t N = A.size(0);

    // Allocate output matrix C on the same device and dtype as A
    auto C = torch::zeros({N, N}, A.options());


     // Determine stream layout: assume NUM_STREAMS is a perfect square
    const int SQRT_STREAMS = static_cast<int>(sqrt(NUM_STREAMS));
    TORCH_CHECK(SQRT_STREAMS * SQRT_STREAMS == NUM_STREAMS, "NUM_STREAMS must be a perfect square");

    // Define block shape (number of threads per block)
    int block_tile_size = N / SQRT_STREAMS;
    dim3 block(TILE, TILE);

     // Create NUM_STREAMS CUDA streams for concurrent execution
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i)
        streams[i] = at::cuda::getStreamFromPool();

    auto A_acc = A.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto B_acc = B.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto C_acc = C.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();

    // Loop over a grid of tiles: each tile assigned to one CUDA stream
    for (int ty = 0; ty < SQRT_STREAMS; ++ty) {
        for (int tx = 0; tx < SQRT_STREAMS; ++tx) {
            // Define the start and end row/column of the tile this stream is responsible for
            int stream_id = ty * SQRT_STREAMS + tx;
            int row_start = ty * block_tile_size;
            int row_end = (ty == SQRT_STREAMS - 1) ? N : (ty + 1) * block_tile_size;

             // Calculate size of the current tile in rows and columns
            int col_start = tx * block_tile_size;
            int col_end = (tx == SQRT_STREAMS - 1) ? N : (tx + 1) * block_tile_size;

            int rows = row_end - row_start;
            int cols = col_end - col_start;
            
            // Launch kernel for this tile asynchronously on its assigned stream
            dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

            matmul_kernel_naive_block_streamed<<<grid, block, 0, streams[stream_id]>>>(
                A_acc, B_acc, C_acc, N, row_start, row_end, col_start, col_end);
        }
    }

    for (auto& stream : streams){cudaStreamSynchronize(stream);}
        

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel launch failed");
    return C;
}
