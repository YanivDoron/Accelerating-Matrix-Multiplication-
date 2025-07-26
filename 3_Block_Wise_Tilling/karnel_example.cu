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

