#ifndef KERNEL_INTERFACE_H
#define KERNEL_INTERFACE_H

#include <cuda_runtime.h>


extern "C" cudaError_t multiply(
    int* C, 
    const int* A, 
    const int* B, 
    size_t sizeA,
    size_t sizeB,
    cudaStream_t stream = nullptr
);


extern "C" const char* getKernelName();

extern "C" const char* getKernelDescription();

#endif // KERNEL_INTERFACE_H 