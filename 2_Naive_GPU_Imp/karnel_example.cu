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

