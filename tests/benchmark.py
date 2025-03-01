#!/usr/bin/env python3
import sys
import argparse
import torch
import triton
from typing import Dict, List
import ctypes

from tests.utils import (
    load_cuda_kernel, get_kernel_paths, get_test_cases,
    load_test_case, run_kernel
)

def benchmark_kernel(lib: ctypes.CDLL, case_name: str, warmup: int = 10, repeat: int = 100) -> float:
    """
    Benchmark a CUDA kernel using triton's do_bench
    
    Args:
        lib: Loaded kernel library
        case_name: Name of test case to use
        warmup: Number of warmup iterations
        repeat: Number of benchmark iterations
        
    Returns:
        Average execution time in milliseconds
    """
    input_a, input_b, _ = load_test_case(case_name)
    
    a_cuda = torch.tensor(input_a, device='cuda', dtype=torch.uint32)
    b_cuda = torch.tensor(input_b, device='cuda', dtype=torch.uint32)
    c_cuda = torch.zeros(len(input_a) + len(input_b), device='cuda', dtype=torch.uint32)
    bigc_cuda = torch.zeros(len(input_a) + len(input_b), device='cuda', dtype=torch.uint64)
    
    a_ptr = ctypes.c_void_p(a_cuda.data_ptr())
    b_ptr = ctypes.c_void_p(b_cuda.data_ptr())
    c_ptr = ctypes.c_void_p(c_cuda.data_ptr())
    bigc_ptr = ctypes.c_void_p(bigc_cuda.data_ptr())
    
    ms = triton.testing.do_bench(
        lambda: lib.multiply(c_ptr, bigc_ptr, a_ptr, b_ptr, len(input_a), len(input_b), None),
        warmup=warmup,
        rep=repeat
    )
    return ms

def run_benchmark(lib: ctypes.CDLL, test_cases: List[str], warmup: int, repeat: int) -> bool:
    """
    Run benchmark on a single kernel
    
    Args:
        lib: Loaded kernel library
        test_cases: List of test cases to benchmark
        warmup: Number of warmup iterations
        repeat: Number of benchmark iterations
        
    Returns:
        True if benchmark completed successfully
    """
    kernel_name = lib.getKernelName().decode('utf-8')
    print(f"Benchmarking: {kernel_name}")
    print(f"Description: {lib.getKernelDescription().decode('utf-8')}")

    success = True

    try:
        for case_name in test_cases:
            print(f"  Testing case: {case_name}")
            try:
                ms = benchmark_kernel(lib, case_name, warmup, repeat)
                print(f"  ✓ {ms:.3f} ms")
                    
            except Exception as e:
                print(f"  ✗ Error benchmarking case {case_name}: {e}")
                success = False
                
    except Exception as e:
        print(f"✗ Error benchmarking kernel: {e}")
        success = False
        
    return success

def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA integer multiplication kernels")
    parser.add_argument("--kernels", nargs='+', help="Paths to kernel libraries to benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--repeat", type=int, default=100, help="Number of benchmark iterations")
    
    args = parser.parse_args()
    
    kernel_paths = get_kernel_paths(args.kernels)
    
    if not kernel_paths:
        print("No kernel libraries found")
        return
    
    print(f"Found {len(kernel_paths)} kernel(s) to benchmark")
    
    test_cases = get_test_cases()
    if not test_cases:
        print("No test cases found")
        return
        
    print(f"Found {len(test_cases)} test case(s)")
    
    all_passed = True
    for kernel_path in kernel_paths:
        try:
            lib = load_cuda_kernel(kernel_path)
            success = run_benchmark(lib, test_cases, args.warmup, args.repeat)
            if not success:
                all_passed = False
                
        except Exception as e:
            print(f"✗ Error loading kernel {kernel_path}: {e}")
            all_passed = False
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main() 