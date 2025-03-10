#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch
import ctypes
from typing import Dict, List, Tuple

from tests.multiply.utils import (
    load_cuda_kernel, get_kernel_paths, get_test_cases,
    load_test_case, run_kernel
)

def uint32_array_to_int(arr: np.ndarray) -> int:
    """Convert array of uint32 back to arbitrary precision integer"""
    result = 0
    for i, val in enumerate(arr):
        unsigned_val = int(val) & 0xFFFFFFFF
        result |= unsigned_val << (32 * i)
    return result

def validate_result(result: np.ndarray, expected: np.ndarray) -> str:
    """
    Validate kernel result against expected value, accounting for trailing zeros

    Args:
        result: Output array from kernel (may contain trailing zeros)
        expected: Expected output array
        
    Returns:
        An empty string if the result is valid, otherwise an error message
    """
    result_int = uint32_array_to_int(result)
    expected_int = uint32_array_to_int(expected)
    
    if result_int == expected_int:
        return ""
    
    error_msg =  f"\n    Expected: {expected_int}"
    error_msg += f"\n    Result: {result_int}"
    error_msg += f"\n    Difference: {result_int - expected_int:+d}"
    
    return error_msg

def validate_kernel(
    lib: ctypes.CDLL, 
    case_names: List[str]
) -> Dict[str, bool]:
    """
    Validate a kernel against multiple test cases
    
    Args:
        lib: Loaded kernel library
        case_names: List of test case names to validate against
        
    Returns:
        Dictionary mapping test case names to validation results
    """
    kernel_name = lib.getKernelName().decode('utf-8')
    print(f"Validating: {kernel_name}")
    
    results = {}
    
    for case_name in case_names:
        print(f"  Testing case: {case_name}")
        
        try:
            input_a, input_b, expected = load_test_case(case_name)
            input_a_int = uint32_array_to_int(input_a)
            input_b_int = uint32_array_to_int(input_b)
            expected_int = uint32_array_to_int(expected)
            
            print(f"    Input A (integer): {input_a_int}")
            print(f"    Input B (integer): {input_b_int}")
            print(f"    Expected (little-endian uint32): {expected}")
            print(f"    Expected (integer): {expected_int}")
            
            result = run_kernel(lib, input_a, input_b)
            result_int = uint32_array_to_int(result)
            print(f"    Result (little-endian uint32): {result}")
            print(f"    Result (integer): {result_int}")
            
            error_msg = validate_result(result, expected)
            
            if error_msg:
                print(f"  ✗ Test case {case_name} failed{error_msg}")
            else:
                print(f"  ✓ Test case {case_name} passed")
            
            results[case_name] = error_msg == ""
            
        except Exception as e:
            print(f"  ✗ Error testing case {case_name}: {e}")
            results[case_name] = False
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Validate CUDA integer multiplication kernels")
    parser.add_argument("--kernels", nargs='+', help="Paths to kernel libraries to validate")
    parser.add_argument("--num-digits", help="Specific digit count subdirectory to test (e.g. '32', '64', '128')")
    
    args = parser.parse_args()
    
    kernel_paths = get_kernel_paths(args.kernels)
    
    if not kernel_paths:
        print("No kernel libraries found")
        return
    
    print(f"Found {len(kernel_paths)} kernel(s) to validate")
    
    test_cases = get_test_cases(args.num_digits)
    if not test_cases:
        digit_info = f" for {args.num_digits} digits" if args.num_digits else ""
        print(f"No test cases found{digit_info}")
        return
    
    print(f"Found {len(test_cases)} test case(s)")
    
    all_passed = True
    for kernel_path in kernel_paths:
        try:
            lib = load_cuda_kernel(kernel_path)
            results = validate_kernel(lib, test_cases)
            
            kernel_passed = all(results.values())
            if kernel_passed:
                print(f"✓ All test cases passed for {os.path.basename(kernel_path)}")
            else:
                print(f"✗ Some test cases failed for {os.path.basename(kernel_path)}")
                all_passed = False
                
        except Exception as e:
            print(f"✗ Error validating kernel {kernel_path}: {e}")
            all_passed = False
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()