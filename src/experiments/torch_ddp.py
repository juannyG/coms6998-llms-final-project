"""
Citation:
* https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html
* https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html
* https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py

torchrun --standalone --nproc_per_node=2 run_experiment.py torch_ddp <CONF_KEY>
"""

import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.synthetic import SyntheticDataset
from tools.metrics.metrics_dataclasses import TrainingResults
from utils.gpu import (
    gpu_memory_allocated,
    gpu_utilization_percent,
    reset_peak_mem,
)
from utils.logger import get_log_file_parent_dir

MODEL_FORWARD_PROFILER_LABEL = "model_forward"
MODEL_LOSS_PROFILER_LABEL = "model_loss"
MODEL_BACKWARD_PROFILER_LABEL = "model_backward"
MODEL_OPTIMIZER_PROFILER_LABEL = "model_optimizer_step"
EXPERIMENT_PROFILER_LABELS = [
    MODEL_FORWARD_PROFILER_LABEL,
    MODEL_LOSS_PROFILER_LABEL,
    MODEL_BACKWARD_PROFILER_LABEL,
    MODEL_OPTIMIZER_PROFILER_LABEL,
    # This label represents the pytorch "all-reduce" primitive that does comms in DDP
    # We can take "loss time" - "all-reduce" time to get "loss compute time"
    "nccl:all_reduce",
]


def run_torch_ddp_experiment(model, conf, device, logger):
    # See: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun
    # create model and move it to GPU with id rank
    ddp_model = None
    if device.type.startswith("cuda"):
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=180))
        torch.cuda.set_device(device)
        model = model.to(device=device, dtype=conf["dtype"])
        ddp_model = DDP(model, device_ids=[device.index])
    else:
        dist.init_process_group(backend="gloo")
        model = model.to(device)
        ddp_model = DDP(model)

    try:
        dataset = SyntheticDataset(
            n_samples=10000, seq_len=conf["seq_len"], vocab_size=conf["vocab_size"]
        )
        sampler = DistributedSampler(dataset)
        loader = DataLoader(
            dataset,
            batch_size=conf["batch_size"] // dist.get_world_size(),
            shuffle=False,  # DistributedSampler is mutually exclusive from shuffle
            num_workers=2,
            pin_memory=True,
            sampler=sampler,
        )
        sampler.set_epoch(0)

        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=conf["lr"])
        criterion = nn.CrossEntropyLoss()

        ddp_model.train()
        step = 0
        total_tokens = 0
        cur_mem = 0
        peak_mem = 0
        gpu_util = 0
        gpu_util_per_step = []
        gpu_mem_per_step = []
        loss = torch.Tensor([0])

        reset_peak_mem()
        t0 = time.perf_counter()
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
            logits = ddp_model(batch)

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
            total_tokens += B * (S - 1)

            cur_mem, peak_mem = gpu_memory_allocated()
            gpu_util = gpu_utilization_percent()
            gpu_mem_per_step.append(cur_mem)
            gpu_util_per_step.append(gpu_util)
            if step % 10 == 0 or step == max_steps - 1:
                logger.info(
                    "Training snapshot",
                    extra={
                        "extra": {
                            "step": f"{step + 1}/{max_steps}",
                            "loss": f"{loss.item():.4f}",
                            "step_time_s": f"{step_time:.4f}",
                            "current_gpu_mem_MB": f"{cur_mem:.1f}",
                            "peak_gpu_mem_MB": f"{peak_mem:.1f}",
                            "gpu_util_percent": gpu_util,
                        }
                    },
                )

        total_time = time.perf_counter() - t0
        total_throughput = total_tokens / total_time

        avg_gpu_mem_mb = (
            sum(gpu_mem_per_step) / len(gpu_mem_per_step) if gpu_mem_per_step else 0
        )
        avg_gpu_util_percent = (
            sum(gpu_util_per_step) / len(gpu_util_per_step) if gpu_util_per_step else 0
        )

        training_results = TrainingResults(
            total_tokens=total_tokens,
            total_time_s=total_time,
            total_throughput=total_throughput,
            final_loss=loss.item(),
            avg_gpu_mem_mb=avg_gpu_mem_mb,
            peak_gpu_mem_mb=peak_mem,
            avg_gpu_util_percent=avg_gpu_util_percent,
        )
        logger.info(
            "Training results",
            extra={"extra": training_results.to_dict()},
        )

        # PROFILER EXAMPLE
        steps = 8
        dir_name = get_log_file_parent_dir(logger)
        worker_name = f"rank_{dist.get_rank()}"
        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=8, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                dir_name, worker_name=worker_name
            ),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
        ) as prof:
            for i in range(steps):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad()

                with record_function(MODEL_FORWARD_PROFILER_LABEL):
                    logits = ddp_model(batch)

                logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                targets = batch[:, 1:].contiguous().view(-1)

                with record_function(MODEL_LOSS_PROFILER_LABEL):
                    loss = F.cross_entropy(logits, targets)
                with record_function(MODEL_BACKWARD_PROFILER_LABEL):
                    loss.backward()
                with record_function(MODEL_OPTIMIZER_PROFILER_LABEL):
                    optimizer.step()
                prof.step()

        if device.type == "cpu":
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    finally:
        dist.destroy_process_group()
