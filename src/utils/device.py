import os
from contextvars import ContextVar

import torch

DEVICE_CTX = ContextVar("device_context", default=None)


def set_device_context(d):
    DEVICE_CTX.set(d)


def get_device(force_cpu=False):
    if DEVICE_CTX.get():
        return DEVICE_CTX.get()

    if force_cpu:
        d = torch.device("cpu")
        set_device_context(d)
        return d

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    d = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    set_device_context(d)
    return d


