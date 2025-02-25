#include "kernel_interface.h"

__global__ void multiplyKernel(
    const uint64_t* a,
    const uint64_t* b,
    const size_t size_a,
    const size_t size_b,
    uint64_t* result
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
}

void multiply(
    const uint64_t* a,
    const uint64_t* b,
    const size_t size_a,
    const size_t size_b,
    uint64_t* result
) {
    const int threadsPerBlock = 256;
    const int blocks = 1;
    
    multiplyKernel<<<blocks, threadsPerBlock>>>(a, b, size_a, size_b, result);
}

extern "C" {
    MultiplyKernelFunc GetMultiplyKernel() {
        return multiply;
    }
}