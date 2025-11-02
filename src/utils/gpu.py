__all__ = [
    "gpu_memory_allocated",
    "reset_peak_mem",
    "gpu_utilization_percent",
]

import torch

from utils.device import get_device

PYNVML_ENABLED = False
try:
    import pynvml

    pynvml.nvmlInit()
    PYNVML_ENABLED = True
except:
    pass


def gpu_memory_allocated():
    # returns current and peak allocated (MB)
    device = get_device()
    if torch.cuda.is_available() and device.type != "cpu":
        cur = torch.cuda.memory_allocated(device) / (1024**2)
        peak = torch.cuda.max_memory_allocated(device) / (1024**2)
        return cur, peak
    return 0.0, 0.0


def reset_peak_mem():
    device = get_device()
    if torch.cuda.is_available() and device.type != "cpu":
        device = get_device()
        torch.cuda.reset_peak_memory_stats(device)


def gpu_utilization_percent():
    device = get_device()
    if PYNVML_ENABLED and torch.cuda.is_available() and device.type != "cpu":
        handle = pynvml.nvmlDeviceGetHandleByIndex(
            device.index if device.index is not None else 0
        )
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu
    return 0
