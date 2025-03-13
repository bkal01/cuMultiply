#include "kernel_interface.h"
#include "host_helpers.h"
#include "device_helpers.h"
#include <stdio.h>
#include <stdint.h>
#include <iostream>
__global__ void multiplyKernel(
    uint32_t* C,
    const uint32_t* A,
    const uint32_t* B,
    size_t sizeA,
    size_t sizeB,
    uint64_t* moduli,
    size_t numModuli,
    uint64_t* W,
    uint32_t* accum,
    uint32_t* temp
) {
    const size_t sizeC = sizeA + sizeB;
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ uint64_t residues[256];
    
    if (idx < numModuli) {
        uint64_t modulus = moduli[idx];

        uint64_t r_A = 0;
        uint64_t r_B = 0;

        for (int i = sizeA; i > 0; i--) {
            r_A = (((unsigned __int128)r_A << 32) % modulus);
            r_A = (r_A + A[i - 1]) % modulus;
        }

        for (int i = sizeB; i > 0; i--) {
            r_B = (((unsigned __int128)r_B << 32) % modulus);
            r_B = (r_B + B[i - 1]) % modulus;
        }
        
        uint64_t product = (uint64_t)(((unsigned __int128)r_A * r_B) % modulus);
        residues[threadIdx.x] = product;
    }
    __syncthreads();

    /*
        Until I figure out how to parallelize this it will just be done on one thread.
        We are using Garner's algorithm here, which means we will convert the product
        from modular form to mixed radix form, then recombine.
    */
    __shared__ uint64_t x[256];
    
    if (idx == 0) {
        for (int i = 0; i < numModuli; i++) {
            x[i] = residues[i];
            for (int j = 0; j < i; j++) {
                uint64_t inverse = W[j * numModuli + i];
                x[i] = (uint64_t)(((unsigned __int128)(x[i] + moduli[i] - x[j]) * inverse) % moduli[i]);
            }
        }
        
        // Initialize C with the first mixed radix digit (x[0])
        C[0] = (uint32_t)(x[0] & 0xFFFFFFFF);
        C[1] = (uint32_t)(x[0] >> 32);
        
        // Set accum to 1 (as a multi-precision integer)
        for (int i = 0; i < sizeC; i++) {
            accum[i] = 0;
            temp[i] = 0;
        }
        accum[0] = 1;
        uint64_t accum_len = 1;
        
        // Recombine the mixed radix representation
        for (int i = 1; i < numModuli; i++) {
            // Multiply accum by previous modulus: accum = accum * moduli[i-1]
            // Use temp as the output buffer to avoid same-array input/output issue
            for (int j = 0; j < sizeC; j++) {
                temp[j] = 0;
            }
            uint64_t temp_len;
            multi_precision_multiply(accum, accum_len, moduli[i-1], temp, &temp_len);
            
            // Copy result back to accum
            for (int j = 0; j < temp_len; j++) {
                accum[j] = temp[j];
                temp[j] = 0;
            }
            accum_len = temp_len;
            
            // Multiply accum by current mixed radix digit: temp = accum * x[i]
            multi_precision_multiply(accum, accum_len, x[i], temp, &temp_len);
            
            // Add temp to C: C += temp
            uint64_t c_len = sizeC;
            multi_precision_add_arrays(C, c_len, temp, temp_len, &c_len);
        }
    }
}


extern "C" cudaError_t multiply(
    uint32_t* C,
    const uint32_t* A,
    const uint32_t* B,
    size_t sizeA,
    size_t sizeB
) {
    /*  
        We pick a set of moduli that are all approximately 2^64, and we want to ensure
        that their product is greater than A * B. The product has at most sizeA + sizeB chunks
        = 32 * (sizeA + sizeB) bits, and each modulus contributes 64 bits to the product of the moduli.
        Therefore, we approximately need ceil((sizeA + sizeB) / 2) moduli.
    */
    uint64_t numModuli = (sizeA + sizeB + 1) / 2;
    uint64_t* moduli = new uint64_t[numModuli];

    uint64_t currentCandidate = UINT64_MAX_PRIME;
    uint64_t count = 0;
    while (count < numModuli) {
        if (isPrime(currentCandidate)) {
            moduli[count] = currentCandidate;
            count++;
        }
        currentCandidate--;
    }

    // Compute the inverse matrix W where W[i][j] is the modular inverse of m_i mod m_j
    uint64_t* W = new uint64_t[numModuli * numModuli];
    for (uint64_t i = 0; i < numModuli; i++) {
        for (uint64_t j = 0; j < numModuli; j++) {
            if (i == j) {
                W[j * numModuli + i] = 1;
            } else if (j < i) {
                W[j * numModuli + i] = modInverse(moduli[j] % moduli[i], moduli[i]);
            }
        }
    }
    uint64_t* d_W;
    cudaMalloc((void**)&d_W, numModuli * numModuli * sizeof(uint64_t));
    cudaMemcpy(d_W, W, numModuli * numModuli * sizeof(uint64_t), cudaMemcpyHostToDevice);
    delete[] W;

    uint64_t* d_Moduli;
    cudaMalloc((void**)&d_Moduli, numModuli * sizeof(uint64_t));
    cudaMemcpy(d_Moduli, moduli, numModuli * sizeof(uint64_t), cudaMemcpyHostToDevice);
    delete[] moduli;

    size_t sizeC = sizeA + sizeB;
    uint32_t* d_accum;
    uint32_t* d_temp;
    cudaMalloc((void**)&d_accum, sizeC * sizeof(uint32_t));
    cudaMalloc((void**)&d_temp, sizeC * sizeof(uint32_t));

    int threadsPerBlock = 256;
    
    // 65535 is the maximum number of blocks that can be used in a CUDA kernel
    int numBlocks = min((numModuli + threadsPerBlock - 1) / threadsPerBlock, (size_t)65535);

    multiplyKernel<<<numBlocks, threadsPerBlock>>>(C, A, B, sizeA, sizeB, d_Moduli, numModuli, d_W, d_accum, d_temp);
    
    cudaFree(d_Moduli);
    cudaFree(d_W);
    cudaFree(d_accum);
    cudaFree(d_temp);
    return cudaGetLastError();
}

extern "C" const char* getKernelName() {
    return "Chinese Remainder Theorem Multiplication";
}

extern "C" const char* getKernelDescription() {
    return "Multiplies two numbers A and B by using the Chinese Remainder Theorem. We compute a set of relatively prime moduli, reduce A and B to their modular representations, multiply the results, then recombine.";
} 