#include "kernel_interface.h"
#include <stdio.h>
#include <stdint.h>


__global__ void multiply_kernel(uint32_t* C, const uint32_t* A, const uint32_t* B, size_t sizeA, size_t sizeB) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        C[0] = A[0] * B[0];
    }
}

extern "C" cudaError_t multiply(
    uint32_t* C,
    const uint32_t* A,
    const uint32_t* B,
    size_t sizeA,
    size_t sizeB
) {
    int numBlocks = 1;
    int threadsPerBlock = 256;
    
    multiply_kernel<<<numBlocks, threadsPerBlock>>>(C, A, B, sizeA, sizeB);
    
    return cudaGetLastError();
}

extern "C" const char* getKernelName() {
    return "Example Multiplication";
}

extern "C" const char* getKernelDescription() {
    return "Multiplies two numbers together. Only works for numbers small enough to fit in a 32-bit integer.";
} 