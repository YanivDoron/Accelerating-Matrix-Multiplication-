import torch
from torch import nn
from torch.autograd import Function

import matmul_cuda  # CUDA extension

torch.manual_seed(42)

# This is the function that will be called from Python
def matmul(A1, A2):
    return matmul_cuda.matmul(A1.contiguous(), A2.contiguous())

def matmul_naive(A1, A2):
    return matmul_cuda.matmul_naive(A1.contiguous(), A2.contiguous()) 

def matmul_naive_block(A1, A2):
    return matmul_cuda.matmul_naive_block(A1.contiguous(), A2.contiguous()) 

def matmul_streamed(A1, A2):
    return matmul_cuda.matmul_streamed(A1.contiguous(), A2.contiguous())

def matmul_block_SM(A1, A2):
    return matmul_cuda.matmul_block_SM(A1.contiguous(), A2.contiguous())

def matmul_streamed_SM(A1, A2):
    return matmul_cuda.matmul_streamed_SM(A1.contiguous(), A2.contiguous())

#Use for REGTILE + SharedMem + Streams DeBugging only!
def matmul_RT_streamed(A1, A2):
    return matmul_cuda.matmul_RT_streamed(A1.contiguous(), A2.contiguous())

def matmul_streamed_tileblock(A1, A2):
    return matmul_cuda.matmul_streamed_tileblock(A1.contiguous(), A2.contiguous())