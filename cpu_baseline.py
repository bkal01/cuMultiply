import math
import random
import statistics
import time

import modal

app = modal.App(name="cuMultiply")

DEFAULT_BIT_LENGTH = 1_000_000
DEFAULT_TRIALS = 5
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

@app.function()
def run_kernel(bit_length: int = DEFAULT_BIT_LENGTH, trials: int = DEFAULT_TRIALS):
    times = []

    for idx in range(1, trials + 1):
        print(f"Trial {idx}/{trials}, bit length: {bit_length}, approx decimal digits: {int(math.log10(2) * bit_length) + 1}")
        start_time = time.time()
        output = random.getrandbits(bit_length) * random.getrandbits(bit_length)
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"Output bit length: {output.bit_length()}, approx decimal digits: {int(math.log10(2) * output.bit_length()) + 1}")
        print(f"Time taken: {round(elapsed, 3)} seconds")

    summary = {
        "mean": statistics.fmean(times) if times else 0.0,
        "min": min(times) if times else 0.0,
        "max": max(times) if times else 0.0,
        "std": statistics.pstdev(times) if len(times) > 1 else 0.0,
    }
    print(f"Timing summary: {summary}")
    return summary

@app.local_entrypoint()
def main(trials: int = DEFAULT_TRIALS, bit_length: int = DEFAULT_BIT_LENGTH):
    run_kernel.remote(bit_length=bit_length, trials=trials)
