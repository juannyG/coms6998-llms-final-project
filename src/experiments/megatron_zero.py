import os
import time
import datetime
from tools.metrics.metrics_dataclasses import TrainingResults
from utils.logger import get_log_file_parent_dir
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import deepspeed

from datasets.synthetic import MegatronSyntheticDataset
from utils.gpu import (
    gpu_memory_allocated,
    gpu_utilization_percent,
    reset_peak_mem,
)

# ---- Profiler labels -----------------
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


def run_megatron_zero_experiment(_, conf, device, logger):
    if device.type == "cpu" or not torch.cuda.is_available():
        print("Megatron experiments cannot be run on CPU devices. Exiting...")
        exit(1)

    """
    This experiment completely ignores the SimpleTransformerDecoder, because we 
    need a Megatron GPTModel instance - which operates completely differently
    than what pytorch offers.

    Furthermore, Megatron assumes you're working with a GPU, so CPU based development
    is a no go.

    Most of this setup and execution comes from
    * https://github.com/NVIDIA/Megatron-LM/blob/main/examples/run_simple_mcore_train_loop.py#L143
    * https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html#write-your-first-training-loop
    """

    parallel_state.destroy_model_parallel()

    # Initialize distributed via DeepSpeed as opposed to megatron...
    deepspeed.init_distributed(
        dist_backend="nccl",
        timeout=datetime.timedelta(seconds=180),
    )
    torch.cuda.set_device(device)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    parallel_state.initialize_model_parallel()  # Defaults tp=pp=1

    try:
        model_parallel_cuda_manual_seed(123)

        tc = TransformerConfig(
            num_layers=conf["n_layers"],
            hidden_size=conf["d_model"],
            num_attention_heads=conf["n_heads"],
        )

        gpt_model = GPTModel(
            config=tc,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=conf["vocab_size"],
            max_sequence_length=conf["seq_len"],
        )
        gpt_model.to(device=device, dtype=conf["dtype"])

        ds_config = {}
        cfg_path = os.environ.get("ZERO_CONFIG")
        if cfg_path:
            with open(cfg_path, "r") as f:
                ds_config = yaml.safe_load(f)
        else:
            print("ERROR: Missing ZERO_CONFIG env var")
            exit(1)

        batch_size = conf["batch_size"]
        ds_config["bf16"] = {"enabled": conf["dtype"] == torch.bfloat16}
        ds_config["train_micro_batch_size_per_gpu"] = batch_size // world_size

        # wrap model with DeepSpeed engine in order to use ZeRO
        optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=conf["lr"])
        model_engine, _, _, _ = deepspeed.initialize(
            model=gpt_model,
            optimizer=optimizer,
            config=ds_config,
        )

        # Dataset / DataLoader (same as megatron_ddp)
        dataset = MegatronSyntheticDataset(
            n_samples=10000,
            seq_len=conf["seq_len"],
            vocab_size=conf["vocab_size"],
        )

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=ds_config["train_micro_batch_size_per_gpu"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            sampler=sampler,
        )
        sampler.set_epoch(0)

        criterion = nn.CrossEntropyLoss()

        model_engine.train()
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

            tokens = batch["tokens"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            labels = batch["labels"].to(device)
            model_engine.zero_grad()

            if device.type.startswith("cuda"):
                torch.cuda.synchronize()
            t_before = time.perf_counter()

            logits = model_engine(tokens, position_ids, attention_mask)
            logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
            targets = labels[:, 1:].contiguous().view(-1)
            loss = criterion(logits, targets)
            model_engine.backward(loss)
            model_engine.step()

            if device.type.startswith("cuda"):
                torch.cuda.synchronize()
            t_after = time.perf_counter()

            # metrics
            step_time = t_after - t_before
            total_tokens += (
                ds_config["train_micro_batch_size_per_gpu"] * conf["seq_len"]
            )

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

        steps = 8
        dir_name = get_log_file_parent_dir(logger)
        worker_name = f"rank_{rank}"
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

                model_engine.zero_grad()
                if device.type.startswith("cuda"):
                    torch.cuda.synchronize()

                tokens = batch["tokens"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                position_ids = batch["position_ids"].to(device)
                labels = batch["labels"].to(device)
                with record_function(MODEL_FORWARD_PROFILER_LABEL):
                    logits = model_engine(tokens, position_ids, attention_mask)

                logits = model_engine(tokens, position_ids, attention_mask)
                logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
                targets = labels[:, 1:].contiguous().view(-1)
                loss = criterion(logits, targets)
                with record_function(MODEL_BACKWARD_PROFILER_LABEL):
                    # all ZeRO comms (reduce-scatter/allgather) happen inside backward/step
                    model_engine.backward(loss)

                with record_function(MODEL_OPTIMIZER_PROFILER_LABEL):
                    model_engine.step()

                if device.type.startswith("cuda"):
                    torch.cuda.synchronize()

                prof.step()

        if device.type == "cpu":
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    finally:
        # finally, clean up process group
        if dist.is_initialized():
            dist.destroy_process_group()
