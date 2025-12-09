import importlib
import math
import random
import statistics
import time

import modal
import torch

app = modal.App(name="cuMultiply")

DEFAULT_BIT_LENGTH = 64
DEFAULT_TRIALS = 5
RANDOM_SEED = 42

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

# convert an integer to a little-endian tensor of uint32 limbs
def int_to_uint32_tensor(value):
    num_limbs = (value.bit_length() + 31) // 32
    limbs = [(value >> (32 * i)) & ((1 << 32) - 1) for i in range(num_limbs)]
    return torch.tensor(limbs, dtype=torch.uint32)


# convert a tensor of uint64 limbs to a Python integer
def uint64_tensor_to_int(limbs):
    val = 0
    for i in range(len(limbs) - 1, -1, -1):
        val = (val << 64) | int(limbs[i].detach().cpu().numpy())
    return val

def normalize(limbs):
    carry = 0
    new_limbs = [0 for _ in range(len(limbs))]
    for i in range(len(limbs)):
        total = limbs[i].item() + carry
        new_limbs[i] = total & ((1 << 32) - 1)
        carry = total >> 32
    while carry:
        new_limbs.append(carry & ((1 << 32) - 1))
        carry >>= 32
    return torch.tensor(new_limbs, dtype=torch.uint32)

def uint32_tensor_to_int(limbs):
    val = 0
    for i in range(len(limbs) - 1, -1, -1):
        val = (val << 32) | int(limbs[i].detach().cpu().numpy())
    return val


def int_to_uint64_tensor(value):
    num_limbs = max(1, (value.bit_length() + 63) // 64)
    limbs = [(value >> (64 * i)) & ((1 << 64) - 1) for i in range(num_limbs)]
    return torch.tensor(limbs, dtype=torch.uint64)


def summarize(times):
    return {
        "mean": statistics.fmean(times),
        "min": min(times),
        "max": max(times),
        "std": statistics.pstdev(times) if len(times) > 1 else 0.0,
    }


random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


@app.function(image=kernel_image, gpu="A100")
def run_kernel(bit_length: int = DEFAULT_BIT_LENGTH, trials: int = DEFAULT_TRIALS, kernel: str = "naive"):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for kernel evaluation.")
    device = torch.device("cuda")
    model = load_kernel_model(kernel).to(device)
    cpu_times = []
    cuda_times = []
    mismatches = 0

    # ensure that the highest bit is set to 1 so that we always have the same number of limbs
    high_bit = 1 << (bit_length - 1)
    for idx in range(1, trials + 1):
        print(f"Trial {idx}/{trials}, bit length: {bit_length}, approx decimal digits: {int(math.log10(2) * bit_length) + 1}")
        a = high_bit | random.getrandbits(bit_length - 1)
        b = high_bit | random.getrandbits(bit_length - 1)

        cpu_start = time.perf_counter()
        cpu_result = a * b
        cpu_elapsed = time.perf_counter() - cpu_start
        cpu_times.append(cpu_elapsed)

        a_tensor = int_to_uint32_tensor(a)
        b_tensor = int_to_uint32_tensor(b)

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start_event.record()
            gpu_output = model(a_tensor, b_tensor)
            end_event.record()

        torch.cuda.synchronize()
        cuda_elapsed = start_event.elapsed_time(end_event) / 1_000
        cuda_times.append(cuda_elapsed)

        gpu_output = gpu_output.to("cpu")
        gpu_output = normalize(gpu_output)
        gpu_result = uint32_tensor_to_int(gpu_output)

        if gpu_result != cpu_result:
            mismatches += 1
            print("Mismatch detected between CPU and CUDA results.")
            print(f"a: {a_tensor}")
            print(f"b: {b_tensor}")
            print(f"CPU result: {int_to_uint32_tensor(cpu_result)}")
            print(f"GPU result: {gpu_output}")

        print(f"CPU time: {cpu_elapsed:.4f}s | CUDA time: {cuda_elapsed:.4f}s | per-trial speedup: {cpu_elapsed / cuda_elapsed if cuda_elapsed else float('inf'):.2f}x")

    cpu_summary = summarize(cpu_times)
    cuda_summary = summarize(cuda_times)
    speedup = cpu_summary["mean"] / cuda_summary["mean"] if cpu_summary["mean"] and cuda_summary["mean"] else float("inf")

    summary = {
        "cpu": cpu_summary,
        "cuda": cuda_summary,
        "speedup": speedup,
        "mismatches": mismatches,
    }

    print(f"CPU timing summary: {cpu_summary}")
    print(f"CUDA timing summary: {cuda_summary}")
    print(f"Mean speedup: {speedup:.2f}x | Mismatches: {mismatches}")
    return summary

@app.local_entrypoint()
def main(trials: int = DEFAULT_TRIALS, bit_length: int = DEFAULT_BIT_LENGTH, kernel: str = "naive"):
    run_kernel.remote(bit_length=bit_length, trials=trials, kernel=kernel)
