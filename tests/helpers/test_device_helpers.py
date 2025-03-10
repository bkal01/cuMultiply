import numpy as np
import torch
import ctypes
import random
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

def uint32_array_to_int(arr):
    """Convert a uint32 array in little endian format to a decimal number"""
    result = 0
    for i in range(len(arr) - 1, -1, -1):
        result = (result << 32) | arr[i]
    return result

def test_multi_precision_add():
    lib = ctypes.CDLL("build/lib/libdevice_helpers_test.so")
    
    lib.launch_multi_precision_add_test.argtypes = [
        ctypes.c_void_p,  # a
        ctypes.c_int,     # a_len
        ctypes.c_uint64,  # addend
        ctypes.c_void_p,  # result
        ctypes.c_void_p,  # result_len
    ]
    
    digit_sizes = [2, 10, 15, 20, 100, 1000]
    
    for digit_size in digit_sizes:
        # Prepare inputs
        bits_needed = int(digit_size * 3.32) + 1  # log2(10) ≈ 3.32
        
        uint32_count = (bits_needed + 31) // 32
        
        num_decimal_digits = random.randint(max(1, digit_size-1), digit_size)
        a_int = random.randint(10**(num_decimal_digits-1), 10**num_decimal_digits - 1)
        
        addend = random.randint(1, 2**64 - 1)
        
        a_array = decimal_to_uint32_array(a_int, min_length=uint32_count)
        a_len = len(a_array)
        
        expected_int = a_int + addend
        expected_array = decimal_to_uint32_array(expected_int)
        expected_len = len(expected_array)
        
        result_buffer_size = max(a_len + 2, expected_len)
        
        a_gpu = torch.from_numpy(a_array).cuda()
        result_gpu = torch.zeros(result_buffer_size, dtype=torch.int32).cuda()
        result_len_gpu = torch.zeros(1, dtype=torch.int64).cuda()
        
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
        
        # Truncate to actual length if provided
        if result_len > 0:
            result_array = result_array[:result_len]
        else:
            # If result_len is not set by the kernel, find the effective length
            # by removing trailing zeros
            for i in range(len(result_array)-1, -1, -1):
                if result_array[i] != 0:
                    result_array = result_array[:i+1]
                    break
        
        result_int = uint32_array_to_int(result_array)
        
        assert result_int == expected_int, f"Test failed for {digit_size} digits:\n" \
                                          f"Input: {a_int} + {addend} = {expected_int}\n" \
                                          f"Got: {result_int}\n" \
                                          f"Expected array: {expected_array}\n" \
                                          f"Result array: {result_array}"
        
        print(f"Multi-precision add test passed for ~{digit_size} decimal digits (actual: {num_decimal_digits})")

def test_device_helpers():
    test_multi_precision_add()

if __name__ == "__main__":
    test_device_helpers()