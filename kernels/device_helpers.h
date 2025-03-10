#ifndef DEVICE_HELPERS_H
#define DEVICE_HELPERS_H

#include <stdint.h>
#include <cuda_runtime.h>

/**
 * Multiplies a multi-precision integer by a 64-bit integer.
 */
__device__ void multi_precision_multiply(const uint32_t *input, size_t length, uint64_t multiplier, uint32_t *result) {
    unsigned __int128 carry = 0;
    size_t i;
    for (i = 0; i < length; i++) {
        unsigned __int128 product = (unsigned __int128)input[i] * multiplier + carry;
        result[i] = (uint32_t)product;      // lower 32 bits
        carry = product >> 32;              // upper bits become new carry
    }
    // Store any remaining carry in two additional 32-bit chunks.
    result[i++] = (uint32_t)carry;
    result[i++] = (uint32_t)(carry >> 32);
}

/**
 * Adds a 64-bit integer to a multi-precision integer.
 */
__device__ void multi_precision_add(const uint32_t *a, int a_len, uint64_t addend, uint32_t *result, uint64_t *result_len) {
    if (a != result) {
        for (int i = 0; i < a_len; i++) {
            result[i] = a[i];
        }
    }
    
    uint64_t carry = addend;
    int i = 0;
    while (carry > 0 && i < a_len) {
        uint64_t sum = (uint64_t)result[i] + carry;
        result[i] = (uint32_t)(sum & 0xFFFFFFFFULL);
        carry = sum >> 32;
        i++;
    }
    
    int len = a_len;
    while (carry > 0) {
        result[len] = (uint32_t)(carry & 0xFFFFFFFFULL);
        carry >>= 32;
        len++;
    }
    *result_len = len;
}

#endif // DEVICE_HELPERS_H 