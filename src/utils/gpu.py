import torch
import pynvml

pynvml.nvmlInit()
"""
TODO: These functions assume 1 GPU.

Needs an update to use torch.cuda.device_count()
with a slight tweak of the contract (i.e. we return
a map of gpu_id -> metric)
"""


def gpu_memory_allocated():
    # returns current and peak allocated (MB)
    if torch.cuda.is_available():
        cur = torch.cuda.memory_allocated() / (1024**2)
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        return cur, peak
    return 0.0, 0.0

def reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def gpu_utilization_percent():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return util.gpu

def compute_throughput(tokens_processed, seconds):
    return tokens_processed / seconds if seconds > 0 else 0.0
