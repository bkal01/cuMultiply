#!/usr/bin/env python3
import ctypes
import os
import numpy as np
import torch
from typing import List, Tuple, Set, Optional

LIB_DIR = "build/lib"
TEST_CASES_DIR = "tests/multiply/test_cases"

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
            ctypes.c_void_p,  # C
            ctypes.c_void_p,  # A
            ctypes.c_void_p,  # B
            ctypes.c_size_t,  # sizeA
            ctypes.c_size_t,  # sizeB
        ]
        lib.multiply.restype = ctypes.c_int
        
        lib.getKernelName.argtypes = []
        lib.getKernelName.restype = ctypes.c_char_p
        
        lib.getKernelDescription.argtypes = []
        lib.getKernelDescription.restype = ctypes.c_char_p
        
        return lib
    except Exception as e:
        print(f"Error loading kernel library {kernel_path}: {e}")
        raise

def get_kernel_paths(specified_kernels=None) -> List[str]:
    """
    Get list of kernel paths either from specified list or by searching LIB_DIR
    
    Args:
        specified_kernels: Optional list of specific kernel paths to use
        
    Returns:
        List of kernel paths
    """
    if specified_kernels:
        return specified_kernels
        
    kernel_paths = []
    for file in os.listdir(LIB_DIR):
        if file.endswith(".so"):
            kernel_paths.append(os.path.join(LIB_DIR, file))
            
    return kernel_paths

def get_test_cases(num_digits: Optional[str] = None) -> List[str]:
    """
    Find all test cases in the test cases directory
    
    Args:
        num_digits: Optional specific digit count subdirectory to look in
        
    Returns:
        Sorted list of test case names
    """
    test_cases = set()
    
    if num_digits:
        subdir = os.path.join(TEST_CASES_DIR, num_digits)
        if os.path.exists(subdir) and os.path.isdir(subdir):
            for file in os.listdir(subdir):
                if file.endswith("_a.bin"):
                    case_name = os.path.join(num_digits, file[:-6])
                    test_cases.add(case_name)
    else:
        for root, dirs, files in os.walk(TEST_CASES_DIR):
            for file in files:
                if file.endswith("_a.bin"):
                    rel_dir = os.path.relpath(root, TEST_CASES_DIR)
                    if rel_dir == ".":
                        case_name = file[:-6]
                    else:
                        case_name = os.path.join(rel_dir, file[:-6])
                    test_cases.add(case_name)
    
    return sorted(list(test_cases))

def load_test_case(case_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a test case from binary files
    
    Args:
        case_name: Name of the test case to load (may include subdirectory)
        
    Returns:
        Tuple of (input_a, input_b, expected_output) numpy arrays
    """
    case_path = os.path.join(TEST_CASES_DIR, case_name)
    
    input_a = np.fromfile(f"{case_path}_a.bin", dtype=np.uint32)
    input_b = np.fromfile(f"{case_path}_b.bin", dtype=np.uint32)
    expected = np.fromfile(f"{case_path}_expected.bin", dtype=np.uint32)
    
    return input_a, input_b, expected

def run_kernel(lib: ctypes.CDLL, input_a: np.ndarray, input_b: np.ndarray) -> np.ndarray:
    """
    Run the kernel with the provided inputs
    
    Args:
        lib: Loaded kernel library
        input_a: First input array
        input_b: Second input array
        
    Returns:
        Output array from the kernel
    """
    a_cuda = torch.tensor(input_a, device='cuda', dtype=torch.uint32)
    b_cuda = torch.tensor(input_b, device='cuda', dtype=torch.uint32)
    c_cuda = torch.zeros(len(input_a) + len(input_b), device='cuda', dtype=torch.uint32)
    
    a_ptr = ctypes.c_void_p(a_cuda.data_ptr())
    b_ptr = ctypes.c_void_p(b_cuda.data_ptr())
    c_ptr = ctypes.c_void_p(c_cuda.data_ptr())
    
    error = lib.multiply(c_ptr, a_ptr, b_ptr, len(input_a), len(input_b), None)
    if error != 0:
        cuda_error = ctypes.c_int(error).value
        raise RuntimeError(f"CUDA error: {cuda_error}")
    
    return c_cuda.cpu().numpy() 