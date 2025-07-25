#include <torch/extension.h>
#include <vector>

// Forward declaration of CUDA kernel (assumes you have it in .cu file)
torch::Tensor matmul_cuda(torch::Tensor A1, torch::Tensor A2);
torch::Tensor matmul_cuda_naive (torch::Tensor A1, torch::Tensor A2);
torch::Tensor matmul_cuda_naive_block(torch::Tensor A1, torch::Tensor A2);
torch::Tensor matmul_cuda_streamed_row(torch::Tensor A1, torch::Tensor A2); // Uncomment if you have a streamed version
torch::Tensor matmul_cuda_streamed_tiled_blocks(torch::Tensor A1, torch::Tensor A2); // Uncomment if you have a streamed version
torch::Tensor matmul_cuda_block_SM(torch::Tensor A1, torch::Tensor A2);
torch::Tensor matmul_cuda_streamed_SM(torch::Tensor A1, torch::Tensor A2);
torch::Tensor matmul_RT_streamed_cuda(torch::Tensor A1, torch::Tensor A2); // Uncomment if you have a streamed version
// Macro checks
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_3D(x) AT_ASSERTM(x.dim() == 3, #x " must be a 3D tensor")
#define CHECK_BATCH_SIZE(x1, x2) AT_ASSERTM(x1.size(0) == x2.size(0), "Batch sizes must match")
#define CHECK_MUL_COMPATIBLE(x1, x2) AT_ASSERTM(x1.size(2) == x2.size(1), "Matrix dimensions must align for multiplication")

#define CHECK_INPUTS(x1, x2) \
    CHECK_CUDA(x1); CHECK_CUDA(x2); \
    CHECK_CONTIGUOUS(x1); CHECK_CONTIGUOUS(x2); \
    CHECK_3D(x1); CHECK_3D(x2); \
    CHECK_BATCH_SIZE(x1, x2); \
    CHECK_MUL_COMPATIBLE(x1, x2)


// C++ interface
// Forward declaration of CUDA kernel (assumes you have it in .cu file)
torch::Tensor matmul(torch::Tensor A1, torch::Tensor A2){
  //CHECK_INPUTS(A1, A2);

  return matmul_cuda(A1, A2);
}

torch::Tensor matmul_naive(torch::Tensor A1, torch::Tensor A2) {
  // Ensure inputs are valid
  return matmul_cuda_naive(A1, A2);

}

torch::Tensor matmul_naive_block(torch::Tensor A1, torch::Tensor A2) {
  // Ensure inputs are valid
  return matmul_cuda_naive_block(A1, A2);
}

torch::Tensor matmul_streamed(torch::Tensor A1, torch::Tensor A2) {
  // Ensure inputs are valid
  return matmul_cuda_streamed_row(A1, A2);
}
torch::Tensor matmul_block_SM(torch::Tensor A1, torch::Tensor A2) {
  // Ensure inputs are valid
  return matmul_cuda_block_SM(A1, A2);
}

torch::Tensor matmul_streamed_SM(torch::Tensor A1, torch::Tensor A2) {
  // Ensure inputs are valid
  return matmul_cuda_streamed_SM(A1, A2);
}

torch::Tensor matmul_RT_streamed(torch::Tensor A1, torch::Tensor A2) {
  
  // Call the CUDA kernel
  return matmul_RT_streamed_cuda(A1, A2);
}

torch::Tensor matmul_streamed_tileblock(torch::Tensor A1, torch::Tensor A2) {
  // Ensure inputs are valid
  return matmul_cuda_streamed_tiled_blocks(A1, A2);
}

// Expose as `matmul` (exact same name!)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul", &matmul, "Matrix Multiplication (CUDA)");
  m.def("matmul_naive", &matmul_naive, "Naive Matrix Multiplication (CUDA)");
  m.def("matmul_naive_block", &matmul_naive_block, "Naive Block Matrix Multiplication (CUDA)");
  m.def("matmul_streamed", &matmul_streamed, "Streamed Matrix Multiplication (CUDA)");
  m.def("matmul_block_SM", &matmul_block_SM, "Block Matrix Multiplication with Shared Memory (CUDA)");
  m.def("matmul_streamed_SM", &matmul_streamed_SM, "Streamed Block Matrix Multiplication with Shared Memory (CUDA)");
  m.def("matmul_RT_streamed", &matmul_RT_streamed, "RT Streamed Matrix Multiplication (CUDA)");
  m.def("matmul_streamed_tileblock",&matmul_streamed_tileblock, "Streamed Matrix Multiplication (CUDA) blocked tile");
}


