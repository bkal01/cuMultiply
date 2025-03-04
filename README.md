# cuMultiply

How fast can we get integer multiplication on an H100?

## Overview

A lot of time and effort has been spent optimizing matmuls on GPUs (see [here](https://siboehm.com/articles/22/CUDA-MMM)). And rightfully, so they're probably the most important
computation to ever be performed on a GPU.

But what about integer multiplication? Multiplying numbers millions of digits long is still pretty important (in cryptography, for example), but it's not being actively optimized
on modern GPUs. At least not publicly.

So, the goal of this repo is to serve as a benchmark for integer multiplication and to try and multiply million digit numbers as fast as we possibly can.

## Problem Statement

We are given two positive numbers $A$ and $B$, each $n$ digits long, and we want to compute $C = A\cdot B$. Since these numbers are too large to fit in a single machine word,
we store them as a sequence of `uint32` chunks in little-endian format. This lets us store and perform operations on numbers of arbitrary length.

## Approaches

### Naive Multiplication

In grade school, we're taught that to multiply two numbers $A$ and $B$ together, we multiply every digit of $A$ with every digit of $B$ and sum them up, accounting for powers of 10.
This approach is unfortunately $O(n^2)$ in time complexity.

In our case, we'll need to multiply every `uint32` chunk of $A$ with every `uint32` chunk of $B$, but the principle and the
complexity are the same.

To distribute this work across threads, we can have each thread compute the product of one `uint32` chunk of $A$ and $B$ and write that an intermediate `uint64` array. Then we can have a single thread perform sequential carry propagation.

For example, if we had a block with just 4 threads, it might look like this:

![image](https://github.com/user-attachments/assets/9371192e-a9ac-427b-97ae-99e56623c255)

While it technically works, and achieves ~ 0.002 seconds for 19-digit multiplication, there are several problems with this approach:
1. $O(n^2)$ is just too slow.
2. We need to use the `atomicAdd` operation, meaning multiple threads needing to accumulate their products to the same chunk need to wait for each other.
3. We have to do carry propagation, which is a sequential operation.
4. It doesn't work for $n \geq 20$ digits! We're accumulating `uint32` products into a `uint64` word. At 20 digits, we'll need two `uint32` chunks to represent $A$ and $B$. Computing one of the chunks of $C$ will require an operation like $A[0]\cdot B[1] + A[1] \cdot B[0]$, which can overflow a `uint64`. Of course we can try to use different and bigger types, maybe 128-bit integers, but that only pushes the problem further down the line.

This is a fundamental problem with the naive multiplication kernel that cannot be fixed unless we incorporate intermediate carry propagation, further reducing the parallel processing here to the point where it doesn't even make sense to use a GPU. So, we move on to better, more interesting multiplication algorithms:

### Chinese Remainder Theorem-based multiplication

Coming soon!

### Schönhage-Strassen Multiplication

Coming soon!
