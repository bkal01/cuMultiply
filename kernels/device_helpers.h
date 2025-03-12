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
__device__ void multi_precision_add(const uint32_t *a, int a_len, uint64_t addend, 
                                      uint32_t *result, uint64_t *result_len) {
    if (a != result) {
        for (int i = 0; i < a_len; i++) {
            result[i] = a[i];
        }
    }
    
    uint32_t low  = (uint32_t)(addend & 0xFFFFFFFF);
    uint32_t high = (uint32_t)(addend >> 32);
    uint64_t carry = 0;

    // Process first two chunks using addend's low and high parts.
    for (int i = 0; i < 2; i++) {
        uint32_t add_val = (i == 0) ? low : high;
        uint64_t limb = (i < a_len ? result[i] : 0);
        uint64_t sum = limb + add_val + carry;
        if (i < a_len)
            result[i] = (uint32_t)sum;
        else {
            result[i] = (uint32_t)sum;
            a_len = i + 1;
        }
        carry = sum >> 32;
    }
    
    // Propagate carry through remaining chunks.
    for (int i = 2; i < a_len && carry; i++) {
        uint64_t sum = (uint64_t)result[i] + carry;
        result[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
    
    // Append a new chunk if carry remains.
    while (carry) {
        result[a_len++] = (uint32_t)carry;
        carry >>= 32;
    }
    
    *result_len = a_len;
}


#endif // DEVICE_HELPERS_H 