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
// --------------------------------------------------- Most Naive Matrix Multiplication Kernel On GPU ---------------------------------------------------------
// ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
// A *very* simple (O(N³) global-memory) GEMM for square matrices.
// Naive matrix multiplication kernel as a reference implementation to GPU-accelerated GEMM.
// This kernel computes C = A * B for square matrices A, B, and C.
// Doesn't use shared memory or any advanced optimizations.
// Each thread computes one element of the output matrix C.

//**Note** : 
// the number of blocks is calculated based on the total number of elements in C divided by the number of threads per block
//This ensures that all elements of C are computed, But it may not be optimal for all cases.
// ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
__global__ void matmul_kernel_naive_flat( torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> A,
                                          torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> B,
                                          torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> C,
                                          const int N){

    // Each thread computes one element of the output matrix C , The thread index is calculated based on the block and thread indices.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;

    int row = idx / N ,  col = idx % N;

    // Each thread computes the dot product of the row of A and the column of B
    float sum = 0;
    for (int k = 0; k < N; ++k) {   sum += A[row][k] * B[k][col];   }

    // Store the result in the output matrix C
    C[row][col] = sum; 
}

// Naive matrix multiplication kernel Launcher function for the flat kernel 
at::Tensor matmul_cuda_naive(const at::Tensor A, const at::Tensor B){

    // Check that the input tensors are on CUDA and have the correct dimensions
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda(),   "Tensors must be on CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,                   "Expected 2-D inputs");
    TORCH_CHECK(A.size(1) == B.size(0),                         "Inner dimensions must match"); 
    
    const int64_t N = A.size(0);                
    auto C = torch::zeros({N, N}, A.options());

    // Set up grid and block dimensions
    // The grid is divided into blocks of size threads 

    const int threads = 1024; //Maximum number of threads per block on GPU Architectures
    const int total = N * N; //the Total number of elements in the output matrix C needed to be computed
    
    // Calculate the number of blocks needed to cover the entire output matrix C
    const int blocks = (total + threads - 1) / threads;

    auto A_acc = A.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto B_acc = B.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto C_acc = C.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();

    matmul_kernel_naive_flat<<<blocks, threads>>>( A_acc, B_acc, C_acc,  static_cast<int>(N));


    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "matmul_kernel_naive_flat launch failed");

    return C;
}


// ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
// -------------------------------------------------------Block-Wise  Matrix Multiplication Kernel On GPU -----------------------------------------------------
//  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
// A naive matrix multiplication kernel that uses a block-wise approach.
// Each block computes a TILE x TILE submatrix of C.
// This kernel is a simple implementation that introduces a block-wise tiling strategy. 

// Fixed block size of 16x16 threads will improve performance over the flat kernel.
//────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
__global__ void matmul_kernel_naive_block(torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> A,
                                          torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> B,
                                          torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> C,
                                          const int N){

    // Each block computes a TILE x TILE submatrix of C
    // The block index is calculated based on the block and thread indices
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    //Check if the thread is within the bounds of the output matrix C
    // If the thread is out of bounds, return early
    if (row > N || col > N) {return; }
    
    // Each thread computes the dot product of the row of A and the column of B
    // The dot product is computed by iterating over the inner dimension N

    float sum = 0;
    for (int k = 0; k < N; ++k) { sum += A[row][k] * B[k][col];}

    // Store the result in the output matrix C
    // Each thread writes its computed value to the global memory
    C[row][col] = sum;

}

// Naive matrix multiplication kernel Launcher function for the block-wise kernel
at::Tensor matmul_cuda_naive_block(const at::Tensor A, const at::Tensor B) {

    // Check that the input tensors are on CUDA and have the correct dimensions
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda(),   "Tensors must be on CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,                   "Expected 2-D inputs");
    TORCH_CHECK(A.size(1) == B.size(0),                         "Inner dimensions must match");

    const int64_t N = A.size(0);               
    auto C = torch::zeros({N, N}, A.options());

    // Set up grid and block dimensions , note that we use a fixed block size of 16x16 threads
    // This block size is a good compromise between occupancy and performance
    // The grid is divided into blocks of size TILE x TILE
    constexpr int TILE = 16;                   // 16×16 threads per block
    const dim3 block(TILE, TILE);

    // The grid is divided into blocks of size TILE x TILE
    const dim3 grid((N + TILE - 1) / TILE,(N + TILE - 1) / TILE);

    auto A_acc = A.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto B_acc = B.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto C_acc = C.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();

    matmul_kernel_naive_block<<<grid, block>>>(A_acc,B_acc, C_acc,static_cast<int>(N));


    // good practice: catch kernel‐launch errors in debug builds
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "matmul_kernel_naive launch failed");

    return C;
}


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


at::Tensor matmul_cuda_streamed(const at::Tensor A, const at::Tensor B) {

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
__global__ void matmul_kernel_block_shared_streamed_fixed(  torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> A,
                                                            torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> B,
                                                            torch::PackedTensorAccessor<float, 2, torch::RestrictPtrTraits, size_t> C,
                                                            int N, int row_offset, int row_limit) {
 
    // Each block computes a TILE x TILE submatrix of C
    // The block index is calculated based on the block and thread indices
    int local_row = blockIdx.y * TILE + threadIdx.y; //add because each stream processes a subset of rows
    int row = local_row + row_offset;
    int col = blockIdx.x * TILE + threadIdx.x;

    //Define shared memory tiles for A and B
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
        As[threadIdx.y][threadIdx.x] = (row < row_limit && tiled_col < N) ? A[row][tiled_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (tiled_row < N && col < N) ? B[tiled_row][col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k){ sum += As[threadIdx.y][k] * Bs[k][threadIdx.x]; }
            
        __syncthreads();
    }

    if (row < row_limit && col < N){C[row][col] = sum; }
        
}

at::Tensor matmul_cuda_streamed_SM(const at::Tensor A, const at::Tensor B) {
    // Check that the input tensors are on CUDA and have the correct dimensions
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda(),   "Tensors must be on CUDA");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,                   "Expected 2-D inputs");
    TORCH_CHECK(A.size(1) == B.size(0),                         "Inner dimensions must match"); 
    TORCH_CHECK(A.scalar_type() == torch::kFloat32,             "Only float32 is supported");

    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    // Set up grid and block dimensions
    dim3 block(TILE, TILE);
    int rows_per_stream = (N + NUM_STREAMS - 1) / NUM_STREAMS;

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; ++i){streams[i] = at::cuda::getStreamFromPool(); }
        

    auto A_acc = A.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto B_acc = B.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();
    auto C_acc = C.packed_accessor<float, 2, torch::RestrictPtrTraits, size_t>();

    for (int s = 0; s < NUM_STREAMS; ++s) {
        // Calculate the start and end rows for this stream with the row_offset and row_limit parameters
        int row_start = s * rows_per_stream;
        int row_end = std::min((s + 1) * rows_per_stream, N);
        int rows_this = row_end - row_start;

        dim3 grid((N + TILE - 1) / TILE, (rows_this + TILE - 1) / TILE);

        matmul_kernel_block_shared_streamed_fixed<<<grid, block, 0, streams[s]>>>(A_acc, B_acc, C_acc, N, row_start, row_end);
            
    }

    for (auto& stream : streams){cudaStreamSynchronize(stream);}
        

    return C;
}

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


at::Tensor matmul_RT_streamed_cuda(torch::Tensor A, torch::Tensor B) {}