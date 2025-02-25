#pragma once

#include <cuda_runtime.h>

typedef void (*MultiplyKernelFunc)(
    const uint64_t* A,
    const uint64_t* B,
    size_t size_A,
    size_t size_B,
    uint64_t* output
);

extern "C" {
    MultiplyKernelFunc GetMultiplyKernel();
}