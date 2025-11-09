import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import deepspeed

from datasets.synthetic import SyntheticDataset
from utils.gpu import(
    gpu_memory_allocated,
    gpu_utilization_percent,
    reset_peak_mem,
)

MODEL_FORWARD_PROFILER_LABEL = "model_forward"
MODEL_LOSS_PROFILER_LABEL = "model_loss"
MODEL_BACKWARD_PROFILER_LABEL = "model_backward"
MODEL_OPTIMIZER_PROFILER_LABEL = "model_optimizer_step"
EXPERIMENT_PROFILER_LABELS = [
    MODEL_FORWARD_PROFILER_LABEL,
    MODEL_LOSS_PROFILER_LABEL,
    MODEL_BACKWARD_PROFILER_LABEL,
    MODEL_OPTIMIZER_PROFILER_LABEL,
]


def _deepspeed_config(zero_stage: int, bf16: bool, offload:bool, grad_accum_steps: int, micro_bs: int):
    cfg = {
        "train_micro_batch_size_per_gpu": micro_bs,
        "gradient_accumulation_steps"   : grad_accum_steps,
        "zero_optimization": {
            "stage": zero_stage,
            "overlap_comm": True,
            "reduce_scatter": True, # for ZeRO 2/3
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7, # tuneable
            "allgather_bucket_size": 5e7, # tuneable
        },
        "bf16": {"enabled": bf16},
        "fp16": {"enabled": False},
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": True, # used to report comm/step timings in DeepSpeed
    }
    if offload:
        # Per the DeepSpeed paper, offloading only changes memory on a single GPU (as it offloads memory to CPU/registers) or when we want more headroom
        cfg["zero_opitimization"].update({
            "offload_param": {"device": "cpu", "pin_memory": True},
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
        })
    return cfg

def run_deepspeed_zero_experiment(model, conf, device, loggger, zero_stage: int, offload: bool=False):

    if device.type.startsWith("cuda"):
        deepspeeed.init_distributed(dist_backend="nccl", timeout=datetime.timedelta(seconds=180))
        torch.cuda.set_device(device)
    else:
        deepspeed.init_distributed(dist_backend="gloo", timeout=datetime.timedelta(seconds=180))

    try:

        micro_bs   = conf["batch_size"]
        grad_accum = conf.get("grad_accum_steps", 1)
        bf16       = conf.get("bf16", torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)

        optimizer = torch.optim.AdamW(model.parameters(), lr=conf["lr"])

        # the code below initializes the DeepSpeed engine running the ZeRO algorithm
        # and returns an engine that replaces the model

        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            model_parameters=model.parameters(),
            config=ds_config,
        )

        dataset=SyntheticDataset(
            n_samples=10000, seq_len=conf["seq_len"], vocab_size=conf["vocab_size"]
        )

        world_size = dist.get_world_size()
        rank       = dist.get_rank()
        sampler    = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

        loader = Dataloader(
            dataset,
            batch_size=micro_bs,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            sampler=sampler,
        )
        sampler.set_epoch(0)

        criterion = torch.nn.CrossEntropyLoss()
        model_engine.train()

        step = total_tokens = 0
        cur_mem = peak_mem = gpu_util = 0
        token_throughputs, sample_throughputs, losses= [], [], []
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

            # move the batch to the engine device
            batch = batch.to(model_engine.local_rank if device.type.startsWith("cuda") else device, non_blocking=True)

            B, S = batch.shape

            # using deepspeed API
            torch.cuda.synchronize() if device.type.startsWith("cuda") else None
            t_before = time.perf_counter()

            logits = model_engine(batch)
            logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            targets = batch[:, 1:].contiguous().view(-1)

            loss = criterion(logits, targets)
            model_engine.backward(loss)
            model_engine.step()

            torch.cuda.synchronize() if device.type.startsWith("cuda") else None
            t_after = time.perf_counter()

            step_time = t_after - t_before
            token = B * (S - 1)
            samples = B
            total_tokens += tokens
            if step >= warmup:
                token_throughputs.append(tokens / step_time)
                sample_throughputs.append(samples / step_time)
                losses.append(loss.item())

            cur_mem, peak_mem = gpu_memory_allocated()
            gpu_util = gpu_utilization_percent()
            if step % 10 == 0 or step == max_steps -1:
                logger.info(
                    "Training snapshot",
                    extra={"extra": {
                        "step": f"{step + 1} / {max_steps}",
                        "loss": f"{loss.item():.4f}",
                        "step_time_s": f"{step_time:.4f}",
                        "tokens_per_s": f"{tokens / step_time:, .0f}",
                        "current_gpu_mem_MB": f"{cur_mem:.1f}",
                        "peak_gpu_mem_MB": f"{peak_mem:.1f}",
                        "gpu_util_percent": gpu_util,
                        "zero_stage": zero_stage,
                        "offload": offload,
                    }}
                )

        total_time = time.perf_counter() - t0
        avg_tokens_s = sum(token_throughputs) / len(token_throughputs)
        avg_samples_s = sum(sample_throughputs) / len(sample_throughputs)
        avg_loss = sum(losses) / len(losses) if losses else None

        logger.info(
            "Training results",
            extra={"extra": {
                "avg_tokens_s": avg_tokens_s,
                "avg_samples_s": avg_samples_s,
                "avg_loss": avg_loss,
                "total_tokens": total_tokens,
                "total_time_s": total_time,
                "cur_gpu_mem_mb": cur_mem,
                "peak_gpu_mem_mb": peak_mem,
                "gpu_util_percent": gpu_util,
                "zero_stage": zero_stage,
                "offload": offload,
            }}
        )

        # PROFILER EXAMPLE
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
                batch = batch.to(model_engine.local_rank if device.type.startsWith("cuda") else device, non_blocking=True)
                optimizer.zero_grad()

                with record_function(MODEL_FORWARD_PROFILER_LABEL):
                    logits = model_engine(batch)

                logtis = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                targets = batch[:, 1:].contiguous().view(-1)

                with record_function(MODEL_LOSS_PROFILER_LABEL):
                    loss = F.cross_entropy(logits, targets)

                with record_function(MODEL_BACKWARD_PROFILER_LABEL):
                    model_engine.backward()
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
        dist.destroy_process_group()





