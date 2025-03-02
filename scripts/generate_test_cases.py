#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import random
from pathlib import Path

OUTPUT_DIR = "tests/test_cases"

def int_to_uint32_array(x: int) -> np.ndarray:
    """Convert arbitrary precision integer to array of uint32"""
    if x == 0:
        return np.array([0], dtype=np.uint32)
    
    chunks = []
    while x > 0:
        chunks.append(x & 0xFFFFFFFF)
        x >>= 32
    return np.array(chunks, dtype=np.uint32)

def generate_test_case(
    num_digits: int,
    case_name: str
) -> None:
    """
    Generate a test case with random integers of specified digit size
    
    Args:
        num_digits: Number of digits in each integer
        case_name: Name for the test case files
    """
    print(f"Generating test case '{case_name}' with {num_digits} digits each")

    # e.g. 2 digits -> 10 to 99.
    min_val = 10**(num_digits-1) if num_digits > 1 else 0
    max_val = 10**num_digits - 1
    
    # We generate random integers with unbounded precision and then convert to arrays of uin32 chunks.
    input_a = random.randint(min_val, max_val)
    input_b = random.randint(min_val, max_val)
    expected = input_a * input_b
    
    input_a_arr = int_to_uint32_array(input_a)
    input_b_arr = int_to_uint32_array(input_b)
    expected_arr = int_to_uint32_array(expected)
    
    digit_dir = os.path.join(OUTPUT_DIR, str(num_digits))
    os.makedirs(digit_dir, exist_ok=True)
    
    input_a_arr.tofile(os.path.join(digit_dir, f"{case_name}_a.bin"))
    input_b_arr.tofile(os.path.join(digit_dir, f"{case_name}_b.bin"))
    expected_arr.tofile(os.path.join(digit_dir, f"{case_name}_expected.bin"))
    
    print(f"  - {num_digits}/{case_name}_a.bin ({input_a_arr.nbytes} bytes)")
    print(f"  - {num_digits}/{case_name}_b.bin ({input_b_arr.nbytes} bytes)")
    print(f"  - {num_digits}/{case_name}_expected.bin ({expected_arr.nbytes} bytes)")

def main():
    parser = argparse.ArgumentParser(description="Generate test cases for CUDA integer multiplication kernels")
    parser.add_argument("--num-cases", type=int, default=5,
                        help="Number of test cases to generate")
    parser.add_argument("--num-digits", type=int, default=3,
                        help="Number of digits in each integer")
    
    args = parser.parse_args()
    print(f"Generating {args.num_cases} test cases with {args.num_digits}-digit integers")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for i in range(args.num_cases):
        generate_test_case(
            num_digits=args.num_digits,
            case_name=f"case_{i+1}"
        )
    
    digit_dir = os.path.join(OUTPUT_DIR, str(args.num_digits))
    print(f"All test cases generated in {digit_dir}")

if __name__ == "__main__":
    main() 