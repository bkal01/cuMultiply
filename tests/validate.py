#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch
import ctypes
from ctypes import c_void_p
from typing import Dict, List, Tuple

from tests.utils import (
    load_cuda_kernel, get_kernel_paths, get_test_cases,
    load_test_case, run_kernel
)

def validate_result(result: np.ndarray, expected: np.ndarray) -> str:
    """
    Validate kernel result against expected value, accounting for trailing zeros

    Args:
        result: Output array from kernel (may contain trailing zeros)
        expected: Expected output array
        
    Returns:
        An empty string if the result is valid, otherwise an error message
    """
    result_last = len(result) - 1
    expected_last = len(expected) - 1
    
    while result_last >= 0 and result[result_last] == 0:
        result_last -= 1
    while expected_last >= 0 and expected[expected_last] == 0:
        expected_last -= 1
        
    if result_last < 0 and expected_last < 0:
        return ""
    
    if result_last == expected_last and np.array_equal(result[:result_last + 1], expected[:expected_last + 1]):
        return ""
    
    max_display = 10
    error_msg = f"\n    Expected (trimmed, showing first {max_display}): {expected[:expected_last + 1][:max_display]}"
    error_msg += f"\n    Result (trimmed, showing first {max_display}): {result[:result_last + 1][:max_display]}"
    
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
            result = run_kernel(lib, input_a, input_b)
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
    
    args = parser.parse_args()
    
    kernel_paths = get_kernel_paths(args.kernels)
    
    if not kernel_paths:
        print("No kernel libraries found")
        return
    
    print(f"Found {len(kernel_paths)} kernel(s) to validate")
    
    test_cases = get_test_cases()
    if not test_cases:
        print("No test cases found")
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