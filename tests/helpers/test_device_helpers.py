import numpy as np
import torch
import ctypes
import random
import argparse
from pathlib import Path

random.seed(42)

def decimal_to_uint32_array(number, min_length=None):
    """Convert a decimal number to uint32 array in little endian format"""
    if number == 0:
        return np.array([0], dtype=np.uint32)
    
    array = []
    while number > 0:
        array.append(number & 0xFFFFFFFF)
        number >>= 32
    
    result = np.array(array, dtype=np.uint32)
    
    if min_length is not None and len(result) < min_length:
        result = np.pad(result, (0, min_length - len(result)), 'constant')
        
    return result

def uint32_array_to_int(arr: np.ndarray) -> int:
    """Convert array of uint32 back to arbitrary precision integer"""
    result = 0
    for i, val in enumerate(arr):
        unsigned_val = int(val) & 0xFFFFFFFF
        result |= unsigned_val << (32 * i)
    return result

def test_multi_precision_add(num_digits_arg=None):
    lib = ctypes.CDLL("build/lib/libdevice_helpers_test.so")
    
    lib.launch_multi_precision_add_test.argtypes = [
        ctypes.c_void_p,  # a
        ctypes.c_int,     # a_len
        ctypes.c_uint64,  # addend
        ctypes.c_void_p,  # result
        ctypes.c_void_p,  # result_len
    ]
    
    digit_sizes = [2, 10, 15, 20, 100, 1000]
    
    if num_digits_arg is not None:
        digit_sizes = [num_digits_arg]
    
    for digit_size in digit_sizes:
        # Prepare inputs
        bits_needed = int(digit_size * 3.32) + 1  # log2(10) ≈ 3.32
        
        uint32_count = (bits_needed + 31) // 32
        
        a_int = random.randint(10**(digit_size-1), 10**digit_size - 1)
        
        addend = random.randint(1, 2**64 - 1)
        
        a_array = decimal_to_uint32_array(a_int, min_length=uint32_count)
        a_len = len(a_array)
        
        expected_int = a_int + addend
        expected_array = decimal_to_uint32_array(expected_int)
        expected_len = len(expected_array)
        
        result_buffer_size = max(a_len + 2, expected_len)
        
        a_gpu = torch.from_numpy(a_array).cuda()
        result_gpu = torch.zeros(result_buffer_size, dtype=torch.uint32).cuda()
        result_len_gpu = torch.zeros(1, dtype=torch.uint64).cuda()
        
        # Call kernel
        lib.launch_multi_precision_add_test(
            a_gpu.data_ptr(),
            a_len,
            int(addend),
            result_gpu.data_ptr(),
            result_len_gpu.data_ptr()
        )
        
        result_array = result_gpu.cpu().numpy()
        result_len = int(result_len_gpu.cpu().numpy()[0])
        
        result_int = uint32_array_to_int(result_array)

        if result_int == expected_int:
            print(f"Multi-precision add test passed for {digit_size} digits")
            print(f"Input: {a_int} + {addend} = {expected_int}")
            print(f"Expected array: {expected_array}")
            print(f"Result array: {result_array}")
        else:
            print(f"Multi-precision add test failed for {digit_size} digits")
            print(f"Input: {a_int} + {addend} = {expected_int}")
            print(f"Got: {result_int}")
            print(f"Expected array: {expected_array}")
            print(f"Result array: {result_array}")

        print("\n" + "="*80 + "\n")

def test_multi_precision_multiply(num_digits_arg=None):
    lib = ctypes.CDLL("build/lib/libdevice_helpers_test.so")
    
    lib.launch_multi_precision_multiply_test.argtypes = [
        ctypes.c_void_p,  # input
        ctypes.c_size_t,  # length
        ctypes.c_uint64,  # multiplier
        ctypes.c_void_p,  # result
        ctypes.c_void_p,  # result_len
    ]
    
    digit_sizes = [2, 10, 15, 20, 100, 1000]
    
    if num_digits_arg is not None:
        digit_sizes = [num_digits_arg]
    
    for digit_size in digit_sizes:
        # Prepare inputs
        bits_needed = int(digit_size * 3.32) + 1  # log2(10) ≈ 3.32
        
        uint32_count = (bits_needed + 31) // 32
        
        input_int = random.randint(10**(digit_size-1), 10**digit_size - 1)
        
        multiplier = random.randint(1, 2**64 - 1)
        
        input_array = decimal_to_uint32_array(input_int, min_length=uint32_count)
        input_len = len(input_array)
        
        expected_int = input_int * multiplier
        expected_array = decimal_to_uint32_array(expected_int)
        expected_len = len(expected_array)
        
        result_buffer_size = input_len + 3
        
        input_gpu = torch.from_numpy(input_array).cuda()
        result_gpu = torch.zeros(result_buffer_size, dtype=torch.uint32).cuda()
        result_len_gpu = torch.zeros(1, dtype=torch.uint64).cuda()
        
        # Call kernel
        lib.launch_multi_precision_multiply_test(
            input_gpu.data_ptr(),
            input_len,
            int(multiplier),
            result_gpu.data_ptr(),
            result_len_gpu.data_ptr()
        )
        
        result_array = result_gpu.cpu().numpy()
        result_len = int(result_len_gpu.cpu().numpy()[0])
        
        result_int = uint32_array_to_int(result_array)

        if result_int == expected_int:
            print(f"Multi-precision multiply test passed for {digit_size} digits")
            print(f"Input: {input_int} * {multiplier} = {expected_int}")
            print(f"Expected array: {expected_array}")
            print(f"Result array: {result_array}")
        else:
            print(f"Multi-precision multiply test failed for {digit_size} digits")
            print(f"Input: {input_int} * {multiplier} = {expected_int}")
            print(f"Got: {result_int}")
            print(f"Expected array: {expected_array}")
            print(f"Result array: {result_array}")

        print("\n" + "="*80 + "\n")

def test_device_helpers(args=None):
    if args:
        if args.ops == "all":
            test_multi_precision_add(num_digits_arg=args.num_digits)
            test_multi_precision_multiply(num_digits_arg=args.num_digits)
        elif args.ops == "add":
            test_multi_precision_add(num_digits_arg=args.num_digits)
        elif args.ops == "multiply":
            test_multi_precision_multiply(num_digits_arg=args.num_digits)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multi-precision math tests')
    parser.add_argument('--ops', type=str, help='Operations to test', default="all")
    parser.add_argument('--num-digits', type=int, help='Specific digit size to test')
    args = parser.parse_args()
    
    test_device_helpers(args)