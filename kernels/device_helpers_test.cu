#include "device_helpers.h"

extern "C" {

__global__ void test_multi_precision_multiply_kernel(
    const uint32_t *input,
    size_t length,
    uint64_t multiplier,
    uint32_t *result
) {
    if (threadIdx.x == 0) {
        multi_precision_multiply(input, length, multiplier, result);
    }
}

__global__ void test_multi_precision_add_kernel(
    const uint32_t *a,
    int a_len,
    uint64_t addend,
    uint32_t *result,
    uint64_t *result_len
) {
    if (threadIdx.x == 0) {
        multi_precision_add(a, a_len, addend, result, result_len);
    }
}

cudaError_t launch_multi_precision_multiply_test(
    const uint32_t *input,
    size_t length,
    uint64_t multiplier,
    uint32_t *result
) {
    test_multi_precision_multiply_kernel<<<1, 32>>>(input, length, multiplier, result);
    return cudaGetLastError();
}

cudaError_t launch_multi_precision_add_test(
    const uint32_t *a,
    int a_len,
    uint64_t addend,
    uint32_t *result,
    uint64_t *result_len
) {
    test_multi_precision_add_kernel<<<1, 32>>>(a, a_len, addend, result, result_len);
    return cudaGetLastError();
}

}