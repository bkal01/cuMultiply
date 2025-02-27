#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
from pathlib import Path

OUTPUT_DIR = "tests/test_cases"

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
    
    input_a = np.random.randint(min_val, max_val, dtype=np.int32)
    input_b = np.random.randint(min_val, max_val, dtype=np.int32)
    expected = input_a * input_b
    
    input_a.tofile(os.path.join(OUTPUT_DIR, f"{case_name}_a.bin"))
    input_b.tofile(os.path.join(OUTPUT_DIR, f"{case_name}_b.bin"))
    expected.tofile(os.path.join(OUTPUT_DIR, f"{case_name}_expected.bin"))
    
    print(f"  - {case_name}_a.bin ({input_a.nbytes} bytes)")
    print(f"  - {case_name}_b.bin ({input_b.nbytes} bytes)")
    print(f"  - {case_name}_expected.bin ({expected.nbytes} bytes)")

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
    
    print(f"All test cases generated in {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 