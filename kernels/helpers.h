#ifndef HELPERS_H
#define HELPERS_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

constexpr uint64_t UINT64_MAX_PRIME = 18446744073709551557ULL;

/**
 * Checks if a uint64_t is prime.
 * Miller-Rabin is deterministic for n < 2^64 with the bases {2, 325, 9375, 28178, 450775, 9780504, 1795265022}.
 */
inline bool isPrime(uint64_t n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0) return false;

    // n - 1 = 2^s * d
    uint64_t d = n - 1;
    int s = 0;
    while ((d & 1) == 0) {
        d >>= 1;
        s++;
    }

    const uint64_t bases[] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};
    const int num_bases = 7;

    for (int i = 0; i < num_bases; i++) {
        uint64_t a = bases[i];
        if (a >= n) a %= n;

        uint64_t x = 1;
        uint64_t p = a;
        uint64_t curr_d = d;

        // n is a strong probable prime to base a if a^d = 1 (mod n) or a^(2^r * d) = -1 (mod n) for some 0 <= r < s.
        while (curr_d > 0) {
            if (curr_d & 1) {
                x = (uint64_t)(((unsigned __int128)x * p) % n);
            }
            p = (uint64_t)(((unsigned __int128)p * p) % n);
            curr_d >>= 1;
        }

        if (x == 1 || x == n - 1) continue;

        bool is_composite = true;
        for (int j = 1; j < s; j++) {
            x = (uint64_t)(((unsigned __int128)x * x) % n);
            if (x == n - 1) {
                is_composite = false;
                break;
            }
        }

        if (is_composite) return false;
    }

    return true;
}

/**
 * Computes the modular inverse of a modulo m using the extended Euclidean algorithm.
 * Returns 0 if the modular inverse doesn't exist.
 */
inline uint64_t modInverse(uint64_t a, uint64_t m) {
    int64_t m0 = m;
    int64_t y = 0, x = 1;
    
    if (m == 1) return 0;
    
    a = a % m;
    
    while (a > 1) {
        int64_t q = a / m;
        int64_t t = m;
        
        m = a % m;
        a = t;
        t = y;
        
        y = x - q * y;
        x = t;
    }
    
    if (x < 0) x += m0;
    
    return x;
}


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


#endif // HELPERS_H 