#include "kernel_interface.h"
#include "helpers.h"
#include <stdio.h>
#include <stdint.h>
#include <iostream>
__global__ void multiplyKernel(
    uint32_t* C,
    uint64_t* bigC,
    const uint32_t* A,
    const uint32_t* B,
    size_t sizeA,
    size_t sizeB,
    uint64_t* moduli,
    size_t numModuli
) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ uint64_t sharedProducts[256];
    __shared__ uint64_t sharedModuli[256];
    
    if (idx < numModuli) {
        uint64_t modulus = moduli[idx];
        sharedModuli[threadIdx.x] = modulus;

        uint64_t r_A = 0;
        uint64_t r_B = 0;

        for (int i = 0; i < sizeA; i++) {
            r_A = (r_A << 32) % modulus;
            r_A = (uint64_t)(((unsigned __int128)r_A + A[i]) % modulus);
        }
        for (int i = 0; i < sizeB; i++) {
            r_B = (r_B << 32) % modulus;
            r_B = (uint64_t)(((unsigned __int128)r_B + B[i]) % modulus);
        }
        
        uint64_t product = (uint64_t)(((unsigned __int128)r_A * r_B) % modulus);
        sharedProducts[threadIdx.x] = product;
    }
    __syncthreads();
}

__global__ void carryPropagationKernel(uint32_t* C, uint64_t* bigC, size_t sizeC) {
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        uint64_t carry = 0;
        for (uint i = 0; i < sizeC; i++) {
            uint64_t total = bigC[i] + carry;
            C[i] = (uint32_t)(total & 0xFFFFFFFF);
            carry = total >> 32;
        }
        if (carry > 0) {
            C[sizeC] = (uint32_t)(carry & 0xFFFFFFFF);
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
            std::cout << "Found prime: " << currentCandidate << "\n";
            moduli[count] = currentCandidate;
            count++;
        }
        currentCandidate--;
    }

    uint64_t* d_Moduli;
    cudaMalloc((void**)&d_Moduli, numModuli * sizeof(uint64_t));
    cudaMemcpy(d_Moduli, moduli, numModuli * sizeof(uint64_t), cudaMemcpyHostToDevice);
    delete[] moduli;

    int threadsPerBlock = 256;
    
    size_t totalWork = sizeA * sizeB;
    // 65535 is the maximum number of blocks that can be used in a CUDA kernel
    int numBlocks = min((totalWork + threadsPerBlock - 1) / threadsPerBlock, MAX_BLOCKS);
    
    size_t sizeC = sizeA + sizeB;

    uint64_t* bigC;
    cudaMalloc((void**)&bigC, sizeC * sizeof(uint64_t));
    cudaMemset(bigC, 0, sizeC * sizeof(uint64_t));
    multiplyKernel<<<numBlocks, threadsPerBlock>>>(C, bigC, A, B, sizeA, sizeB, d_Moduli, numModuli);
    carryPropagationKernel<<<1, threadsPerBlock>>>(C, bigC, sizeC);
    
    cudaFree(bigC);
    cudaFree(d_Moduli);
    return cudaGetLastError();
}

extern "C" const char* getKernelName() {
    return "Chinese Remainder Theorem Multiplication";
}

extern "C" const char* getKernelDescription() {
    return "Multiplies two numbers A and B by using the Chinese Remainder Theorem. We compute a set of relatively prime moduli, reduce A and B to their modular representations, multiply the results, then recombine.";
} 