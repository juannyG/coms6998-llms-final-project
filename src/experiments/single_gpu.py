"""
python run_experiment single_gpu <CONF_KEY>
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader

from datasets.synthetic import SyntheticDataset
from utils.gpu import (
    gpu_memory_allocated,
    gpu_utilization_percent,
    reset_peak_mem,
)


def run_single_gpu_experiment(model, conf, device, logger):
    # TODO: This won't work for multi-GPU setups - diff strats have diff wrappers
    # Things like the rank and flags indicating types of experiment will likely be necessary
    dataset = SyntheticDataset(
        n_samples=10000, seq_len=conf["seq_len"], vocab_size=conf["vocab_size"]
    )
    # TODO: This won't work for multi-GPU strats, we need to include a "sampler"
    loader = DataLoader(
        dataset,
        batch_size=conf["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=conf["lr"])
    criterion = nn.CrossEntropyLoss()

    model.train()
    step = total_tokens = 0
    cur_mem = peak_mem = gpu_util = 0
    token_throughputs = []
    sample_throughputs = []
    losses = []
    reset_peak_mem()
    t0 = time.perf_counter()
    warmup = conf["warmup_steps"]
    max_steps = conf["max_steps"]
    it = iter(loader)
    for step in range(max_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = batch.to(device, non_blocking=True)  # (B, S)
        B, S = batch.shape
        optimizer.zero_grad()

        # forward
        torch.cuda.synchronize() if device.type.startswith("cuda") else None
        t_before = time.perf_counter()
        logits = model(batch)

        # shift logits/targets for next-token prediction
        logits = (
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
        )  # (B*(S-1), V)
        targets = batch[:, 1:].contiguous().view(-1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize() if device.type.startswith("cuda") else None
        t_after = time.perf_counter()

        # metrics
        step_time = t_after - t_before
        tokens = B * (S - 1)  # tokens processed for training step
        samples = B
        total_tokens += tokens
        if step >= warmup:
            token_throughputs.append(tokens / step_time)
            sample_throughputs.append(samples / step_time)
            losses.append(loss.item())

        cur_mem, peak_mem = gpu_memory_allocated()
        gpu_util = gpu_utilization_percent()
        if step % 10 == 0 or step == max_steps - 1:
            logger.info(
                "Training snapshot",
                extra={
                    "extra": {
                        "step": f"{step + 1}/{max_steps}",
                        "loss": f"{loss.item():.4f}",
                        "step_time_s": f"{step_time:.4f}",
                        "tokens_per_s": f"{tokens / step_time:,.0f}",
                        "current_gpu_mem_MB": f"{cur_mem:.1f}",
                        "peak_gpu_mem_MB": f"{peak_mem:.1f}",
                        "gpu_util_percent": gpu_util,
                    }
                },
            )

    total_time = time.perf_counter() - t0
    avg_tokens_per_s = (
        sum(token_throughputs) / len(token_throughputs) if token_throughputs else 0
    )
    avg_samples_per_s = (
        sum(sample_throughputs) / len(sample_throughputs) if sample_throughputs else 0
    )
    avg_loss = sum(losses) / len(losses) if losses else None

    logger.info(
        "Training results",
        extra={
            "extra": {
                "avg_tokens_per_s": avg_tokens_per_s,
                "avg_samples_per_s": avg_samples_per_s,
                "avg_loss": avg_loss,
                "total_tokens": total_tokens,
                "total_time_s": total_time,
                "cur_gpu_mem_mb": cur_mem,
                "peak_gpu_mem_mb": peak_mem,
                "gpu_util_percent": gpu_util,
            }
        },
    )

    # PROFILER EXAMPLE
    # TODO: Modify to support multi-GPU setups
    steps = 8
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for i in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()

            with record_function("model_forward"):
                logits = model(batch)

            logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            targets = batch[:, 1:].contiguous().view(-1)

            with record_function("model_loss"):
                loss = F.cross_entropy(logits, targets)

            with record_function("model_backward"):
                loss.backward()
            with record_function("model_optimizer_step"):
                optimizer.step()

    profiler_metrics = {
        "profiler_metrics": [
            {
                "operation": k.key,
                "count": k.count,
                "cpu_memory_usage": k.cpu_memory_usage,
                "cpu_time_total": k.cpu_time_total,
                "device_memory_usage": k.device_memory_usage,
                "device_time_total": k.device_time_total,
                "device_type": "CUDA" if k.device_time_total > 0 else "CPU",
                "self_cpu_memory_usage": k.self_cpu_memory_usage,
                "self_cpu_time_total": k.self_cpu_time_total,
                "self_device_time_total": k.self_device_time_total,
                "self_device_memory_usage": k.self_device_memory_usage,
            }
            for k in prof.key_averages()
        ]
    }
    logger.info("Profiler metrics", extra={"extra": profiler_metrics})
