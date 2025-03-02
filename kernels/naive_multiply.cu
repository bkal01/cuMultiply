#include "kernel_interface.h"
#include <stdio.h>
#include <stdint.h>
/*
    Multiplies two numbers A and B by computing the product of each "digit" of A and B
    and then adding them together.
*/
__global__ void multiplyKernel(uint32_t* C, uint64_t* bigC, const uint32_t* A, const uint32_t* B, size_t sizeA, size_t sizeB) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint stride = blockDim.x * gridDim.x;
    for (uint i = idx; i < sizeA * sizeB; i += stride) {
        int aPos = i / sizeB;
        int bPos = i % sizeB;
        if (aPos < sizeA && bPos < sizeB) {
            uint64_t product = (uint64_t)A[aPos] * (uint64_t)B[bPos];
            int cPos = aPos + bPos;
            // cast as unsigned long long int to match atomicAdd signature.
            atomicAdd((unsigned long long int*)bigC + cPos, (unsigned long long int)product);
        }
    }
}

__global__ void carryPropagationKernel(uint32_t* C, uint64_t* bigC, size_t sizeC) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        uint64_t carry = 0;
        for (uint i = 0; i < sizeC; i++) {
            uint64_t total = bigC[i] + carry;
            C[i] = (uint32_t)(total & 0xFFFFFFFF);
            carry = total >> 32;
        }
        if (carry > 0) {
            C[sizeC] = (uint32_t)(carry & 0xFFFFFFFF);
        }
    }
}
extern "C" cudaError_t multiply(
    uint32_t* C,
    const uint32_t* A,
    const uint32_t* B,
    size_t sizeA,
    size_t sizeB
) {
    int threadsPerBlock = 256;
    
    size_t totalWork = sizeA * sizeB;
    // 65535 is the maximum number of blocks that can be used in a CUDA kernel
    int numBlocks = min((totalWork + threadsPerBlock - 1) / threadsPerBlock, MAX_BLOCKS);
    
    size_t sizeC = sizeA + sizeB;

    uint64_t* bigC;
    cudaMalloc((void**)&bigC, sizeC * sizeof(uint64_t));
    cudaMemset(bigC, 0, sizeC * sizeof(uint64_t));
    multiplyKernel<<<numBlocks, threadsPerBlock>>>(C, bigC, A, B, sizeA, sizeB);
    carryPropagationKernel<<<1, threadsPerBlock>>>(C, bigC, sizeC);
    
    cudaFree(bigC);
    return cudaGetLastError();
}

extern "C" const char* getKernelName() {
    return "Naive Multiplication";
}

extern "C" const char* getKernelDescription() {
    return "Multiplies two numbers A and B by iterating over each digit of A and B and multiplying them together.";
} 