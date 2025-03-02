#ifndef KERNEL_INTERFACE_H
#define KERNEL_INTERFACE_H

#include <cuda_runtime.h>
#include <stdint.h>

#define MAX_BLOCKS ((size_t)65535)

extern "C" cudaError_t multiply(
    uint32_t* C,
    uint64_t* bigC,
    const uint32_t* A, 
    const uint32_t* B, 
    size_t sizeA,
    size_t sizeB,
    cudaStream_t stream = nullptr
);


extern "C" const char* getKernelName();

extern "C" const char* getKernelDescription();

#endif // KERNEL_INTERFACE_H 