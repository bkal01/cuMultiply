import importlib
import math
import random
import statistics
import time

import modal
import torch

app = modal.App(name="cuMultiply")

DEFAULT_BIT_LENGTH = 10_000_000
DEFAULT_TRIALS = 5
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

kernel_image = (
    modal.Image.from_registry(
        f"nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.10",
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.9.0",
        "ninja",
        "numpy",
    )
    .add_local_dir(
        local_path="kernels",
        remote_path="/root/kernels",
    )
)

def load_kernel_model(kernel_name):
    spec = importlib.util.spec_from_file_location(kernel_name, f"kernels/{kernel_name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_cls = getattr(module, "ModelNew", None) or getattr(module, "Model", None)
    if not model_cls:
        raise ValueError(f"No model found in {kernel_name}.py")
    return model_cls().eval()

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


@app.function(image=kernel_image, gpu="A100")
async def run_kernel(warmup_input, inputs, kernel: str = "naive"):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for kernel evaluation.")
    device = torch.device("cuda")
    model = load_kernel_model(kernel).to(device)
    results = []
    elapsed_times = []

    a_warm, b_warm = warmup_input
    a_warm_tensor, b_warm_tensor = torch.frombuffer(bytearray(a_warm), dtype=torch.uint32).to(device), torch.frombuffer(bytearray(b_warm), dtype=torch.uint32).to(device)
    with torch.no_grad():
        for _ in range(3):
            model(a_warm_tensor, b_warm_tensor)
            torch.cuda.synchronize()

    for i, (a, b) in enumerate(inputs):
        print(f"Executing trial {(i+1)}/{len(inputs)} on GPU...")
        a_tensor, b_tensor = torch.frombuffer(bytearray(a), dtype=torch.uint32), torch.frombuffer(bytearray(b), dtype=torch.uint32)
        a_tensor, b_tensor = a_tensor.to(device), b_tensor.to(device)

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start_event.record()
            gpu_output = model(a_tensor, b_tensor)
            end_event.record()

        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event) / 1_000
        print(f"Time taken: {elapsed:.6f} seconds")
        elapsed_times.append(elapsed)

        gpu_output = gpu_output.to("cpu")
        gpu_output = normalize(gpu_output)

        results.append(gpu_output.numpy().tobytes())

    return results, elapsed_times

@app.local_entrypoint()
def main(trials: int = DEFAULT_TRIALS, bit_length: int = DEFAULT_BIT_LENGTH, kernel: str = "naive"):
    from eval_utils import gu32ops

    start = time.perf_counter()
    inputs, gpu_inputs = generate_inputs(bit_length, trials)
    _, gpu_warmup_inputs = generate_inputs(bit_length, 1)
    gpu_warmup_input = gpu_warmup_inputs[0]

    # we fire off the Modal function to run in the background while we do the CPU computation.
    # even though CPU execution time is higher (at higher bit lengths), we don't want to do nothing during the Modal startup time.
    modal_function_call = run_kernel.spawn(warmup_input=gpu_warmup_input, inputs=gpu_inputs, kernel=kernel)
    cpu_results, cpu_elapsed_times = cpu_baseline(inputs)
    gpu_results, gpu_elapsed_times = modal_function_call.get()

    passed_trials = 0

    for i in range(trials):
        cpu_result, cpu_elapsed_time, gpu_result, gpu_elapsed_time = cpu_results[i], cpu_elapsed_times[i], gpu_results[i], gpu_elapsed_times[i]
        gpu_result = int.from_bytes(gpu_result, "little")
        if cpu_result != gpu_result:
            print(f"Mismatch detected between CPU and GPU results for trial {i+1}")
            print(f"CPU result: {cpu_result}")
            print(f"GPU result: {gpu_result}")
        else:
            passed_trials += 1
            speedup = cpu_elapsed_time / gpu_elapsed_time
            print(f"Passed trial {i+1}")
            print(f"CPU time: {cpu_elapsed_time:.6f}s | GPU time: {gpu_elapsed_time:.6f}s | speedup: {speedup:.2f}x")
            print(f"Gu32ops/sec: {gu32ops(bit_length, kernel) / gpu_elapsed_time}")

    print(f"Passed {passed_trials}/{trials} trials")
    print(f"CPU summary: {summarize(cpu_elapsed_times)}")
    print(f"GPU summary: {summarize(gpu_elapsed_times)}")

    wall_clock_time = time.perf_counter() - start
    print(f"Total time taken to execute {trials} trials: {wall_clock_time:.2f} seconds")
