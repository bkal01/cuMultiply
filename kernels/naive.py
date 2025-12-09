import random

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

bigint_mul_cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE 16

__global__ void bigint_mul_kernel(const uint* __restrict__ A,
                              const uint* __restrict__ B,
                              uint64_t* __restrict__ C,
                              int M) {
    __shared__ uint As[TILE];
    __shared__ uint Bs[TILE];
    __shared__ uint64_t Cs[2*TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int a_tile_index = blockIdx.y * TILE;
    int b_tile_index = blockIdx.x * TILE;

    if (ty == 0) {
        if (a_tile_index + tx < M) {
            As[tx] = A[a_tile_index + tx];
        } else {
            As[tx] = 0;
        }
        if (b_tile_index + tx < M) {
            Bs[tx] = B[b_tile_index + tx];
        } else {
            Bs[tx] = 0;
        }
        Cs[tx] = 0;
        Cs[tx + TILE] = 0;
    }
    __syncthreads();

    if (a_tile_index + tx + b_tile_index + ty < 2*M) {
        unsigned long long int prod = ((unsigned long long int)As[ty]) * ((unsigned long long int)Bs[tx]);
        atomicAdd((unsigned long long int*)&Cs[tx + ty], prod);
    }
    __syncthreads();

    int tid = tx * TILE + ty;
    if (tid < 2 * TILE && a_tile_index + b_tile_index + tid < 2*M) {
        atomicAdd((unsigned long long int*)&C[a_tile_index + b_tile_index + tid], (unsigned long long int)Cs[tid]);
    }
}

torch::Tensor bigint_mul_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure inputs are uint32 and move to CUDA (if not already)
    if (A.dtype() != torch::kUInt32) {
        A = A.to(torch::kUInt32);
    }
    if (B.dtype() != torch::kUInt32) {
        B = B.to(torch::kUInt32);
    }

    if (!A.is_cuda()) {
        A = A.contiguous().to(torch::kCUDA);
    } else {
        A = A.contiguous();
    }
    if (!B.is_cuda()) {
        B = B.contiguous().to(torch::kCUDA);
    } else {
        B = B.contiguous();
    }

    if (A.size(0) != B.size(0)) {
         throw std::runtime_error("bigint_mul_cuda: A and B must have the same number of elements");
    }

    int M = A.size(0);
    auto options = torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCUDA);
    torch::Tensor C = torch::zeros({2 * M}, options);

    dim3 threads(TILE, TILE);
    int tiles = (M + TILE - 1) / TILE;
    dim3 blocks(tiles, tiles);

    bigint_mul_kernel<<<blocks, threads>>>(A.data_ptr<uint32_t>(), B.data_ptr<uint32_t>(), C.data_ptr<uint64_t>(), M);

    return C;
}
"""

bigint_mul_cuda_cpp = "torch::Tensor bigint_mul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile inline extension
bigint_mul_module = load_inline(
    name="naive_bigint_mul",
    cpp_sources=bigint_mul_cuda_cpp,
    cuda_sources=bigint_mul_cuda_source,
    functions=["bigint_mul_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Naive multiplication kernel using a custom CUDA kernel.
    """
    def __init__(self) -> None:
        super().__init__()
        self.bigint_mul = bigint_mul_module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.bigint_mul.bigint_mul_cuda(A, B)

# Keep the same input-generation helpers as the original architecture
N = 1_000

def get_inputs():
    A = random.getrandbits(N)
    B = random.getrandbits(N)

    num_limbs = (N + 32 - 1) // 32
    mask = (1 << 32) - 1
    A_limbs = []
    for _ in range(num_limbs):
        A_limbs.append(A & mask)
        A >>= 32
    B_limbs = []
    for _ in range(num_limbs):
        B_limbs.append(B & mask)
        B >>= 32
    return [torch.tensor(A_limbs, dtype=torch.uint32), torch.tensor(B_limbs, dtype=torch.uint32)]

def get_init_inputs():
    return []