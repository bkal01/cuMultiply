#include "kernel_interface.h"
#include <stdio.h>


__global__ void multiply_kernel(int* C, const int* A, const int* B, size_t sizeA, size_t sizeB) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        C[0] = -1;
    }
}

extern "C" cudaError_t multiply(
    int* C, 
    const int* A, 
    const int* B, 
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
    return "Incorrect Multiplication";
}

extern "C" const char* getKernelDescription() {
    return "Incorrectly sets the result to be -1.";
} 