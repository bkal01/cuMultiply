import math
import random
import statistics
import time

import torch

def gu32ops(bit_length, kernel):
    """
    compute the number of Giga uint32 operations required to multiply two numbers of the given bit length & kernel method
    """
    if kernel == "naive":
        return math.ceil(bit_length / 32)**2 / 1e9

# convert an integer to a bytes, represented as a little-endian array of uint32 limbs
def int_to_uint32_bytes(value):
    num_bytes = math.ceil(value.bit_length() / 8)
    b = value.to_bytes(num_bytes, "little")
    b += b'\x00' * (4 - len(b) % 4)
    return b

def normalize(limbs):
    """
    we have a limbs tensor that look like this:
    [limb_0_lane_0, limb_0_lane_1, limb_1_lane_0, ...]
    with each lane being a uint64 and pairs of lanes representing high/low bits of a "uint128" limb.
    we want to normalize this to a tensor of uint32s.
    """
    carry = 0
    new_limbs = []
    for i in range(0, len(limbs), 2):
        low, high = limbs[i], limbs[i + 1]
        total = low.item() + (high.item() << 64) + carry
        new_limbs.append(total & ((1 << 32) - 1))
        carry = total >> 32
    while carry:
        new_limbs.append(carry & ((1 << 32) - 1))
        carry >>= 32
    return torch.tensor(new_limbs, dtype=torch.uint32)

def summarize(times):
    return {
        "mean": statistics.fmean(times),
        "min": min(times),
        "max": max(times),
        "std": statistics.pstdev(times) if len(times) > 1 else 0.0,
    }

def generate_inputs(bit_length, trials):
    inputs = []
    gpu_inputs = []
    high_bit = 1 << (bit_length - 1)
    for _ in range(trials):
        a = high_bit | random.getrandbits(bit_length - 1)
        b = high_bit | random.getrandbits(bit_length - 1)
        inputs.append((a, b))
        gpu_inputs.append((int_to_uint32_bytes(a), int_to_uint32_bytes(b)))
    return inputs, gpu_inputs

def cpu_baseline(inputs):
    """
    compute CPU baseline for a list of (a, b) pairs.
    """
    results = []
    elapsed_times = []
    for i, (a, b) in enumerate(inputs):
        print(f"Executing trial {(i+1)}/{len(inputs)} on CPU...")
        start_time = time.perf_counter()
        result = a * b
        elapsed = time.perf_counter() - start_time
        print(f"Time taken: {elapsed:.6f} seconds")
        results.append(result)
        elapsed_times.append(elapsed)
    return results, elapsed_times