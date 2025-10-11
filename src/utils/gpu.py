__all__ = [
    'gpu_memory_allocated',
    'reset_peak_mem',
    'gpu_utilization_percent',
    'compute_throughput'
]

from collections import defaultdict

import torch
import pynvml

pynvml.nvmlInit()


def get_gpu_id(i):
    return f'gpu_{i}'

def gpu_memory_allocated():
    # returns current and peak allocated (MB)
    gpu_mem_alloc = {get_gpu_id(0): {'cur': 0.0, 'peak': 0.0}}
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            gpu_mem_alloc[get_gpu_id(i)] = {
                'cur': torch.cuda.memory_allocated() / (1024**2),
                'peak': torch.cuda.max_memory_allocated() / (1024**2),
            }
    return gpu_mem_alloc

def reset_peak_mem():
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.reset_peak_memory_stats()

def gpu_utilization_percent():
    util_perc = {get_gpu_id(0): 0}
    for i in range(torch.cuda.device_count()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        util_perc[get_gpu_id(i)] = util.gpu
    return util_perc

def compute_throughput(tokens_processed, seconds):
    return tokens_processed / seconds if seconds > 0 else 0.0
