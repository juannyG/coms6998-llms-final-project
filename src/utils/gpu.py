__all__ = [
    'gpu_memory_allocated',
    'reset_peak_mem',
    'gpu_utilization_percent',
    'compute_throughput'
]

import torch
import pynvml

pynvml.nvmlInit()


"""
TODO: These functions assume 1 GPU.

* Need to update to become aware of local_rank
* Need to write metrics to a file
* SLURM job will (should?) dump all files back executed during job
* Separate "post-train" script can aggregate cross-device metrics
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
