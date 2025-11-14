"""
Citations:
* https://docs.pytorch.org/docs/2.9/distributed.pipelining.html#module-torch.distributed.pipelining.stage
* https://github.com/pytorch/PiPPy/blob/main/examples/huggingface/pippy_gpt2.py

GPipe experiment (PyTorch built-in, alpha)

Usage:
  torchrun --standalone --nproc_per_node=<NUM_STAGES> run_experiment.py torch_gpipe <CONF_KEY>

Notes:
- One process per pipeline stage.
- Rank 0 owns the input loader; the last rank computes the next-token loss.
- We broadcast the token batch from rank 0 to the last rank each step (only those two ranks participate).
"""

import datetime
import time

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

from datasets.synthetic import SyntheticDataset
from utils.gpu import (
    gpu_memory_allocated,
    gpu_utilization_percent,
    reset_peak_mem,
)

EXPERIMENT_PROFILER_LABELS = []


def run_torch_gpipe_experiment(model, conf, device, logger):
    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=180))
    else:
        dist.init_process_group(backend="gloo", timeout=datetime.timedelta(seconds=180))

    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        model.to(device)

        if world_size == 1:
            raise Exception(
                f"Even on CPU, --nproc_per_node must be > 1: given {world_size}"
            )

        if conf["n_layers"] < world_size:
            raise Exception(
                f"Model layers must be >= world_size: {conf['n_layers']} is less than {world_size}"
            )

        # TODO: make it configurable, but for now, use ideal n_microbatches = 2 * world_size
        # NOTE: we use world_size as stage count, but more sophisticated setups can mave multiple stages per device
        n_microbatches = 2 * world_size
        if n_microbatches < world_size:
            raise Exception(
                f"Number of microbatches must be >= stage count aka world_size: {n_microbatches} is less than {world_size}"
            )

        """
        Build the "split_spec" for pytorch's automatic models splitter.
        This tells `pipeline` how to build the new model graph, which also
        makes it extremely easy to schedule the stage in the pipeline.

        This split spec evenly distributes the layers across all the devices we have.
        An example helps me keep this straight:

        With world_size=3, n_layers=12, we get decoders_per_rank=4:
        i=1: 'layers.4': SplitPoint.BEGINNING  # Start GPU 1 at layer 4
        i=2: 'layers.8': SplitPoint.BEGINNING  # Start GPU 2 at layer 8

        GPU 0 gets: layers 0, 3 (everything before first split)
        GPU 1 gets: layers 4, 7 (from layers.4 to next split)  
        GPU 2 gets: layers 8, 11 (from layers.8 to end)
        """
        decoders_per_rank = (conf["n_layers"] + world_size - 1) // world_size
        split_spec = {
            f"layers.{i * decoders_per_rank}": SplitPoint.BEGINNING
            for i in range(1, world_size)
        }

        # NOTE: Because gpipe uses microbatches of the batch_size, the tensor given to the forward function is batch_size // n_microbatches
        # We also need a "starter" tensor for the pipeline to understand what the "type" of arguments it will receive.
        # The `pipeline` method is
        gpipe_batch_size = conf["batch_size"] // n_microbatches
        init_tensor = torch.randint(
            0, conf["vocab_size"], (gpipe_batch_size, conf["seq_len"])
        )
        pipe = pipeline(
            module=model,
            mb_args=(init_tensor,),
            split_spec=split_spec,
        )
        # print(f"[{rank}]", pipe)

        """
        For simplicity, each device/rank will be a stage. Gpipe can support more sophosticated staging, but
        we're not getting into that...not yet at least.
        """
        stage = pipe.build_stage(rank, device)
        schedule = ScheduleGPipe(stage, n_microbatches)

        dataset = SyntheticDataset(
            n_samples=10000, seq_len=conf["seq_len"], vocab_size=conf["vocab_size"]
        )
        # TODO: Double check num_workers + pin_memory in the other loaders for consistency or annote why it's different
        loader = DataLoader(
            dataset,
            batch_size=conf["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        # NOTE: We only initialize the optimizer with the stage's parameters
        optimizer = torch.optim.AdamW(stage.submod.parameters(), lr=conf["lr"])
        # TODO: rename `criterion` elsewhere...it's just the loss fn...don't be fancy
        loss_fn = nn.CrossEntropyLoss()

        step = 0
        total_tokens = 0
        cur_mem = 0
        peak_mem = 0
        gpu_util = 0
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
            B, S = batch.shape
            optimizer.zero_grad()

            """
            Forward & backward pass, via the pipeline.

            Coordination in the pipeline is enforced by being blocked at send/recv. For example:
            rank-0 here divvies up sending out the microbatches and is blocked until the batch is 
            completely processed by the pipeline, then it does it's optimizer step and preps the 
            second batch.
            """
            loss = torch.Tensor([0])
            t_before = time.perf_counter()
            if rank == 0:
                # First stage sends input
                schedule.step(batch.to(device))
            elif rank == world_size - 1:
                # Last stage receives output and calculates loss
                logits = schedule.step()
                logits = (
                    logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                )  # (B*(S-1), V)
                targets = batch[:, 1:].contiguous().view(-1)

                # Reshape for loss calculation: (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
                loss = loss_fn(logits.to(device), targets.to(device))

                # Backward pass starts from last stage
                loss.backward()

                if step % 10 == 0:
                    # TODO: Remove - debugging purposes
                    print(f"[rank-{rank}] Step {step}, Loss: {loss.item()}")
            else:
                # Intermediate stages just pass data through
                schedule.step()

            # Update parameters
            optimizer.step()
            t_after = time.perf_counter()
            step_time = t_after - t_before

            cur_mem, peak_mem = gpu_memory_allocated()
            gpu_util = gpu_utilization_percent()
            tokens = 0
            samples = 0
            total_tokens = 0
            if rank == world_size - 1:
                # When the last rank gets here, the whole model has seen the whole batch across all layers
                tokens = B * (S - 1)  # tokens processed for training step
                samples = B
                total_tokens += tokens
                if step >= warmup:
                    token_throughputs.append(tokens / step_time)
                    sample_throughputs.append(samples / step_time)
                    losses.append(loss.item())

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
        avg_tokens_per_s = 0
        avg_samples_per_s = 0
        avg_loss = 0
        if rank == world_size - 1:
            # TODO: Explain why we're only using last rank for these metrics
            avg_tokens_per_s = (
                sum(token_throughputs) / len(token_throughputs) if token_throughputs else 0
            )
            avg_samples_per_s = (
                sum(sample_throughputs) / len(sample_throughputs)
                if sample_throughputs
                else 0
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
        # TODO: Profiler loop
        steps = 8
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_stack=False,
        ) as prof:
            pass
    finally:
        dist.barrier()
        dist.destroy_process_group()
