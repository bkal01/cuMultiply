#include "kernel_interface.h"
#include <stdio.h>
#include <stdint.h>
/*

*/
__global__ void multiply_kernel(uint32_t* C, const uint32_t* A, const uint32_t* B, size_t sizeA, size_t sizeB) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint i = idx; i < sizeA * sizeB; i += blockDim.x) {
        int aPos = i % sizeA;
        int bPos = i / sizeA;
        if (aPos < sizeA && bPos < sizeB) {
            unsigned long long product = (unsigned long long)A[aPos] * B[bPos];
            unsigned long long carry = product;
            int cPos = aPos + bPos;
            while (carry > 0) {
                unsigned long long old_val = atomicAdd(&C[cPos], (uint32_t)(carry & 0xFFFFFFFF));
            }
        }
    }
}

extern "C" cudaError_t multiply(
    uint32_t* C,
    const uint32_t* A,
    const uint32_t* B,
    size_t sizeA,
    size_t sizeB,
    cudaStream_t stream
) {
    int numBlocks = 1;
    int threadsPerBlock = 256;
    
    multiply_kernel<<<numBlocks, threadsPerBlock>>>(C, A, B, sizeA, sizeB);
    
    return cudaGetLastError();
}

extern "C" const char* getKernelName() {
    return "Naive Multiplication";
}

extern "C" const char* getKernelDescription() {
    return "Multiplies two numbers A and B by iterating over each digit of A and B and multiplying them together.";
} 