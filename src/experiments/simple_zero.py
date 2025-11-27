import os
import time
import datetime
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import deepspeed

from datasets.synthetic import SyntheticDataset
from utils.gpu import (
    gpu_memory_allocated,
    gpu_utilization_percent,
    reset_peak_mem,
)

# ---- Profiler labels -----------------
MODEL_FORWARD_PROFILER_LABEL = "model_forward"
MODEL_LOSS_PROFILER_LABEL = "model_loss"
MODEL_BACKWARD_PROFILER_LABEL = "model_backward"
MODEL_DDP_PROFILER_LABEL = "zero_communication"
MODEL_OPTIMIZER_PROFILER_LABEL = "model_optimizer_step"
EXPERIMENT_PROFILER_LABELS = [
    MODEL_FORWARD_PROFILER_LABEL,
    MODEL_LOSS_PROFILER_LABEL,
    MODEL_BACKWARD_PROFILER_LABEL,
    MODEL_DDP_PROFILER_LABEL,
    MODEL_OPTIMIZER_PROFILER_LABEL,
]


def _fallback_ds_config(zero_stage: int, bf16: bool, offload: bool,
                        grad_accum_steps: int, micro_bs: int):
    """
    Basic DeepSpeed ZeRO config used only if ZERO_CONFIG is not set.

    For proper runs, refer to src/zero_configs/

    """
    cfg = {
        "train_micro_batch_size_per_gpu": micro_bs,
        "gradient_accumulation_steps": grad_accum_steps,
        "zero_optimization": {
            "stage": zero_stage,
            "overlap_comm": True,
            "reduce_scatter": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,
            "allgather_bucket_size": 5e7,
        },
        "bf16": {"enabled": bf16},
        "fp16": {"enabled": False},
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": True,
    }

    if offload:
        cfg["zero_optimization"].update(
            {
                "offload_param": {"device": "cpu", "pin_memory": True},
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
            }
        )

    return cfg


def run_zero_experiment(model, conf, device, logger, zero_stage: int = 1, offload: bool = False):
    """
    DeepSpeed ZeRO experiment.

    - Stage and offload are *defaults*; if ZERO_CONFIG is set, the YAML
      completely controls the DeepSpeed config.
    - Called from run_experiment.py via EXPERIMENT_TYPES["zero"].
    """

    # ---- Initialize distributed via DeepSpeed ----
    if device.type.startswith("cuda"):
        deepspeed.init_distributed(
            dist_backend="nccl",
            timeout=datetime.timedelta(seconds=180),
        )
        torch.cuda.set_device(device)
    else:
        deepspeed.init_distributed(
            dist_backend="gloo",
            timeout=datetime.timedelta(seconds=180),
        )

    try:
        # build DeepSpeed config (YAML override if present)
        micro_bs = conf["batch_size"]
        grad_accum = conf.get("grad_accum_steps", 1)
        # only auto-enable bf16 on Ampere+ if user didn't set it
        if "bf16" in conf:
            bf16 = bool(conf["bf16"])
        else:
            bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

        cfg_path = os.environ.get("ZERO_CONFIG")
        if cfg_path:
            with open(cfg_path, "r") as f:
                ds_config = yaml.safe_load(f)
        else:
            ds_config = _fallback_ds_config(
                zero_stage=zero_stage,
                bf16=bf16,
                offload=offload,
                grad_accum_steps=grad_accum,
                micro_bs=micro_bs,
            )

        # wrap model with DeepSpeed engine in order to use ZeRO
        optimizer = torch.optim.AdamW(model.parameters(), lr=conf["lr"])
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            model_parameters=model.parameters(),
            config=ds_config,
        )

        # Dataset / DataLoader (same SyntheticDataset as DDP)
        dataset = SyntheticDataset(
            n_samples=10000,
            seq_len=conf["seq_len"],
            vocab_size=conf["vocab_size"],
        )

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=micro_bs,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            sampler=sampler,
        )
        sampler.set_epoch(0)

        criterion = nn.CrossEntropyLoss()

        model_engine.train()
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

            batch = batch.to(device, non_blocking=True)
            B, S = batch.shape

            model_engine.zero_grad()

            if device.type.startswith("cuda"):
                torch.cuda.synchronize()
            t_before = time.perf_counter()

            logits = model_engine(batch)  # (B, S, V)
            logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            targets = batch[:, 1:].contiguous().view(-1)

            loss = criterion(logits, targets)
            model_engine.backward(loss)
            model_engine.step()

            if device.type.startswith("cuda"):
                torch.cuda.synchronize()
            t_after = time.perf_counter()

            # metrics
            step_time = t_after - t_before
            tokens = B * (S - 1)
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
                            "zero_stage": ds_config.get("zero_optimization", {}).get("stage", zero_stage),
                            "offload": bool(
                                ds_config.get("zero_optimization", {}).get("offload_param")
                                or ds_config.get("zero_optimization", {}).get("offload_optimizer")
                            ),
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

        # profiler pass
        steps = 8
        activities = [ProfilerActivity.CPU]
        if device.type.startswith("cuda"):
            activities.append(ProfilerActivity.CUDA)

        with profile(
            activities=activities,
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
                model_engine.zero_grad()

                with record_function(MODEL_FORWARD_PROFILER_LABEL):
                    logits = model_engine(batch)
                logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                targets = batch[:, 1:].contiguous().view(-1)

                with record_function(MODEL_LOSS_PROFILER_LABEL):
                    loss = F.cross_entropy(logits, targets)

                with record_function(MODEL_BACKWARD_PROFILER_LABEL):
                    # all ZeRO comms (reduce-scatter/allgather) happen inside backward/step
                    with record_function(MODEL_DDP_PROFILER_LABEL):
                        model_engine.backward(loss)

                with record_function(MODEL_OPTIMIZER_PROFILER_LABEL):
                    model_engine.step()

        profiler_metrics = {
            "profiler_metrics": [
                {
                    "operation": k.key,
                    "count": k.count,
                    "cpu_memory_usage": k.cpu_memory_usage,
                    "cpu_time_total": k.cpu_time_total,
                    "device_memory_usage": k.device_memory_usage,
                    "device_time_total": k.device_time_total,
                    "device_type": str(k.device_type),
                    "self_cpu_memory_usage": k.self_cpu_memory_usage,
                    "self_cpu_time_total": k.self_cpu_time_total,
                    "self_device_time_total": k.self_device_time_total,
                    "self_device_memory_usage": k.self_device_memory_usage,
                }
                for k in prof.key_averages()
            ]
        }
        logger.info("Profiler metrics", extra={"extra": profiler_metrics})

        if device.type == "cpu":
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    finally:
        # finally, clean up process group
        if dist.is_initialized():
            dist.destroy_process_group()

