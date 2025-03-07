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

#endif // HELPERS_H 