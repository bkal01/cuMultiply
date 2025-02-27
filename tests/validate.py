#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch
import ctypes
from ctypes import c_int, c_size_t, c_void_p, c_char_p
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path

LIB_DIR = "build/lib"
TEST_CASES_DIR = "tests/test_cases"

def load_cuda_kernel(kernel_path: str) -> ctypes.CDLL:
    """
    Load a CUDA kernel library and set the function signatures
    
    Args:
        kernel_path: Path to the compiled .so file
        
    Returns:
        Loaded library with configured function signatures
    """
    try:
        lib = ctypes.CDLL(kernel_path)
        
        # Set function signatures for the library
        lib.multiply.argtypes = [
            c_void_p,  # C
            c_void_p,  # A
            c_void_p,  # B
            c_size_t,  # sizeA
            c_size_t,  # sizeB
            c_void_p   # stream
        ]
        lib.multiply.restype = c_int
        
        lib.getKernelName.argtypes = []
        lib.getKernelName.restype = c_char_p
        
        lib.getKernelDescription.argtypes = []
        lib.getKernelDescription.restype = c_char_p
        
        return lib
    except Exception as e:
        print(f"Error loading kernel library {kernel_path}: {e}")
        raise

def load_test_case(test_case_dir: str, case_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a test case from binary files
    
    Args:
        test_case_dir: Directory containing test cases
        case_name: Name of the test case to load
        
    Returns:
        Tuple of (input_a, input_b, expected_output) numpy arrays
    """
    input_a = np.fromfile(os.path.join(test_case_dir, f"{case_name}_a.bin"), dtype=np.uint32)
    input_b = np.fromfile(os.path.join(test_case_dir, f"{case_name}_b.bin"), dtype=np.uint32)
    expected = np.fromfile(os.path.join(test_case_dir, f"{case_name}_expected.bin"), dtype=np.uint32)
    
    return input_a, input_b, expected

def run_kernel(lib: ctypes.CDLL, input_a: np.ndarray, input_b: np.ndarray) -> np.ndarray:
    """
    Run the kernel with the provided inputs
    
    Args:
        lib: Loaded kernel library
        input_a: First input integer
        input_b: Second input integer
        
    Returns:
        Output integer from the kernel
    """
    a_cuda = torch.tensor(input_a, device='cuda', dtype=torch.int32)
    b_cuda = torch.tensor(input_b, device='cuda', dtype=torch.int32)
    c_cuda = torch.zeros(len(input_a) + len(input_b), device='cuda', dtype=torch.int32)
    
    a_ptr = c_void_p(a_cuda.data_ptr())
    b_ptr = c_void_p(b_cuda.data_ptr())
    c_ptr = c_void_p(c_cuda.data_ptr())
    
    # Run the kernel with size 1
    error = lib.multiply(c_ptr, a_ptr, b_ptr, len(input_a), len(input_b), None)
    if error != 0:
        cuda_error = ctypes.c_int(error).value
        raise RuntimeError(f"CUDA error: {cuda_error}")
    
    # Copy result back to CPU and return the single value
    result = c_cuda.cpu().numpy()
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
    test_case_dir: str, 
    case_names: List[str]
) -> Dict[str, bool]:
    """
    Validate a kernel against multiple test cases
    
    Args:
        lib: Loaded kernel library
        test_case_dir: Directory containing test cases
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
            input_a, input_b, expected = load_test_case(test_case_dir, case_name)
            
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
    
    if args.kernels:
        kernel_paths = args.kernels
    else:
        kernel_paths = []
        for file in os.listdir(LIB_DIR):
            if file.endswith(".so"):
                kernel_paths.append(os.path.join(LIB_DIR, file))
    
    if not kernel_paths:
        print(f"No kernel libraries found in {LIB_DIR}")
        return
    
    print(f"Found {len(kernel_paths)} kernel(s) to validate")
    
    test_cases = set()
    for file in os.listdir(TEST_CASES_DIR):
        if file.endswith("_a.bin"):
            case_name = file[:-6]
            test_cases.add(case_name)
    
    test_cases = sorted(list(test_cases))
    if not test_cases:
        print(f"No test cases found in {TEST_CASES_DIR}")
        return
    
    print(f"Found {len(test_cases)} test case(s)")
    
    all_passed = True
    for kernel_path in kernel_paths:
        try:
            lib = load_cuda_kernel(kernel_path)
            results = validate_kernel(lib, TEST_CASES_DIR, test_cases)
            
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