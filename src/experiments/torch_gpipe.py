# experiments/torch_gpipe.py
"""
GPipe experiment (PyTorch built-in, alpha)

Usage:
  torchrun --standalone --nproc_per_node=<NUM_STAGES> run_experiment.py torch_gpipe <CONF_KEY>

Notes:
- One process per pipeline stage.
- Rank 0 owns the input loader; the last rank computes the next-token loss.
- We broadcast the token batch from rank 0 to the last rank each step (only those two ranks participate).
"""

import datetime
import os
import time
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

from datasets.synthetic import SyntheticDataset
from utils.gpu import (
    gpu_memory_allocated,
    gpu_utilization_percent,
    reset_peak_mem,
)

# ---- profiler labels (match single_gpu) ----
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

# ---------------------------
# Partition wrapper (no model edits needed)
# ---------------------------

class DecoderStageModule(nn.Module):
    """
    Wraps your SimpleTransformerDecoder to execute only:
      - (optional) embedding + dropout + pos emb (stage 0)
      - a contiguous slice of decoder blocks [start_block, end_block) (middle stages)
      - (optional) final layer norm + head (last stage)

    Blocks are numbered globally as 0..(n_layers-1).
    """
    def __init__(
        self,
        full_model: nn.Module,
        n_layers: int,
        start_block: int,
        end_block: int,
        include_embed: bool,
        include_head: bool,
    ):
        super().__init__()
        # Keep references to the original submodules (no copy)
        self.tok_emb = full_model.tok_emb if include_embed else None
        self.pos_emb = full_model.pos_emb if include_embed else None
        self.drop = full_model.drop if include_embed else None

        # We hold only the needed block slice as a ModuleList (references)
        self.blocks = nn.ModuleList()
        if n_layers > 0:
            for i in range(start_block, end_block):
                self.blocks.append(full_model.layers[i])

        self.ln_f = full_model.ln_f if include_head else None
        self.head = full_model.head if include_head else None

    @staticmethod
    def _causal_mask(seq_len: int, device, dtype):
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype), diagonal=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Contract:
          - On stage 0, x is token IDs (B,S) and we embed -> (B,S,D).
          - On middle/last stages, x is already the hidden (B,S,D).
          - On the last stage, we return logits (B,S,V). Otherwise we return hidden (B,S,D).
        """
        if self.tok_emb is not None:
            # x is tokens (B,S) here
            B, S = x.shape
            x = self.tok_emb(x) + self.pos_emb[:, :S, :]
            x = self.drop(x)
        else:
            # x is hidden states
            B, S, _ = x.shape

        # Run our local block slice (pre-LN decoder with MHA)
        for layer in self.blocks:
            attn_ln = layer["ln1"](x)
            attn_mask = self._causal_mask(S, attn_ln.device, attn_ln.dtype)
            attn_out, _ = layer["attn"](attn_ln, attn_ln, attn_ln, attn_mask=attn_mask)
            x = x + attn_out
            ff_ln = layer["ln2"](x)
            x = x + layer["ff"](ff_ln)

        # If we own the head, produce logits
        if self.head is not None:
            x = self.ln_f(x)
            x = self.head(x)  # (B,S,V)
        return x


# ---------------------------
# Partitioning helpers
# ---------------------------

def compute_balance(n_layers: int, n_stages: int) -> List[int]:
    """
    Returns a count of modules per stage in terms of:
      [embed] + [block x ?] + ... + [block x ?] + [head]
    where the sum equals 1 + n_layers + 1.
    """
    if n_stages == 1:
        return [1 + n_layers + 1]

    mid_stages = max(n_stages - 2, 1)
    base = n_layers // mid_stages
    extra = n_layers % mid_stages

    balance = [1]
    for i in range(mid_stages):
        balance.append(base + (1 if i < extra else 0))
    balance.append(1)
    return balance


def stage_block_range(balance: List[int], rank: int, n_layers: int) -> Tuple[int, int, bool, bool]:
    """
    Given a balance vector and a rank, return:
      start_block (inclusive), end_block (exclusive), include_embed, include_head
    Blocks are indexed 0..n_layers-1. Embed sits before block 0; head sits after block n_layers-1.
    """
    # Turn module counts into prefix ranges over [embed][blocks...][head]
    starts = [0]
    for c in balance[:-1]:
        starts.append(starts[-1] + c)
    ends = [s + c for s, c in zip(starts, balance)]

    s, e = starts[rank], ends[rank]
    include_embed = (s == 0)
    include_head = (e == 1 + n_layers + 1)

    # Map module indices to block indices: module 1..n_layers correspond to blocks 0..n_layers-1
    start_block = max(0, s - 1)
    end_block = min(n_layers, e - 1)
    return start_block, end_block, include_embed, include_head


# ---------------------------
# Loss for next-token prediction (applied on last stage)
# ---------------------------

def tokenwise_loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    outputs: (B,S,V) from the last stage
    targets: (B,S) int64 tokens (we'll compute next-token loss by shifting)
    """
    V = outputs.size(-1)
    logits = outputs[:, :-1, :].reshape(-1, V)
    t = targets[:, 1:].reshape(-1)
    return F.cross_entropy(logits, t)


# ---------------------------
# Main experiment
# ---------------------------

def _init_dist(device):
    # Pick backend based on device
    if device.type.startswith("cuda"):
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=180))
        torch.cuda.set_device(device)
    else:
        dist.init_process_group(backend="gloo", timeout=datetime.timedelta(seconds=180))

def run_torch_gpipe_experiment(model, conf, device, logger):
    """
    Entry point (signature matches your other experiments).

    Requirements:
      - Launch with torchrun (one process per stage).
      - conf must include: seq_len, vocab_size, batch_size, lr, warmup_steps, max_steps, n_layers, n_microbatches
    """
    # Require torchrun; provide a clear error if not used
    if "WORLD_SIZE" not in os.environ or "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError(
            "torch_gpipe requires torchrun. Example:\n"
            "  torchrun --standalone --nproc_per_node=2 run_experiment.py torch_gpipe cpu"
        )
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))

    _init_dist(device)

    try:
        # Form a group for broadcasting labels from rank 0 -> last rank only
        last_rank = world - 1
        loss_group = dist.new_group(ranks=[0, last_rank])

        # Dataset / loader (only rank 0 needs a loader)
        dataset = SyntheticDataset(
            n_samples=10000, seq_len=conf["seq_len"], vocab_size=conf["vocab_size"]
        )
        if rank == 0:
            loader = DataLoader(
                dataset,
                batch_size=conf["batch_size"],
                shuffle=True,
                num_workers=2,
                pin_memory=True,
            )
            it = iter(loader)
        else:
            it = None  # no data loader on non-zero ranks

        # Partition the model for this rank
        n_layers = conf["n_layers"]
        balance = compute_balance(n_layers, world)
        sb, eb, inc_embed, inc_head = stage_block_range(balance, rank, n_layers)
        stage_module = DecoderStageModule(
            full_model=model, n_layers=n_layers, start_block=sb, end_block=eb,
            include_embed=inc_embed, include_head=inc_head
        ).to(device)

        # Wrap in PipelineStage
        stage = PipelineStage(
            submodule=stage_module,
            stage_index=rank,
            num_stages=world,
            device=device,
        )

        # Per-stage optimizer
        optim = torch.optim.AdamW(stage.submodule.parameters(), lr=conf["lr"])

        # GPipe schedule (sync SGD semantics)
        schedule = ScheduleGPipe(
            stage,
            n_microbatches=conf.get("n_microbatches", 8),
            loss_fn=tokenwise_loss_fn
        )

        # Training loop
        warmup = conf["warmup_steps"]
        max_steps = conf["max_steps"]
        model.train()
        reset_peak_mem()

        cur_mem = peak_mem = 0.0
        gpu_util = 0
        token_tputs, sample_tputs, losses = [], [], []
        total_tokens = 0
        t0 = time.perf_counter()

        for step in range(max_steps):
            # Rank 0 prepares a batch of token IDs
            if rank == 0:
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                batch = batch.to(device, non_blocking=True)
                B, S = batch.shape

            # Only the LAST rank needs labels (targets) to compute the loss.
            # Rank 0 sends the *same token tensor* as the target labels.
            if rank == 0:
                targets_for_last = batch
                dist.broadcast(targets_for_last, src=0, group=loss_group)
                targets_last = None
            elif rank == last_rank:
                targets_last = torch.empty(
                    (conf["batch_size"], conf["seq_len"]), dtype=torch.long, device=device
                )
                dist.broadcast(targets_last, src=0, group=loss_group)  # receive
            else:
                targets_last = None  # middle ranks don't need labels

            # Timers (rank 0)
            if device.type.startswith("cuda") and rank == 0:
                torch.cuda.synchronize()
            if rank == 0:
                t_before = time.perf_counter()

            # One GPipe training step across all micro-batches
            if rank == 0:
                schedule.step(batch)  # feeds inputs
            elif rank == last_rank:
                schedule.step(target=targets_last, losses=losses)  # computes loss/backward on last stage
            else:
                schedule.step()  # run pipeline

            # Optimizer step per-stage
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Timers (rank 0)
            if device.type.startswith("cuda") and rank == 0:
                torch.cuda.synchronize()
            if rank == 0:
                t_after = time.perf_counter()
                step_time = t_after - t_before
                tokens = conf["batch_size"] * (conf["seq_len"] - 1)
                samples = conf["batch_size"]
                total_tokens += tokens
                if step >= warmup:
                    token_tputs.append(tokens / step_time)
                    sample_tputs.append(samples / step_time)
                if torch.cuda.is_available():
                    cur_mem, peak_mem = gpu_memory_allocated()
                    gpu_util = gpu_utilization_percent()

                if step % 10 == 0 or step == max_steps - 1:
                    logger.info(
                        "Training snapshot (GPipe)",
                        extra={"extra": {
                            "step": f"{step + 1}/{max_steps}",
                            "tokens_per_s": f"{(token_tputs[-1] if token_tputs else 0):,.0f}",
                            "current_gpu_mem_MB": f"{cur_mem:.1f}",
                            "peak_gpu_mem_MB": f"{peak_mem:.1f}",
                            "gpu_util_percent": gpu_util,
                            "balance": balance,
                            "stage_block_range": [sb, eb],
                        }}
                    )

        # Final metrics
        if rank == 0:
            total_time = time.perf_counter() - t0
            avg_tps = (sum(token_tputs)/len(token_tputs)) if token_tputs else 0
            avg_sps = (sum(sample_tputs)/len(sample_tputs)) if sample_tputs else 0
            logger.info(
                "Training results (GPipe)",
                extra={"extra": {
                    "avg_tokens_per_s": avg_tps,
                    "avg_samples_per_s": avg_sps,
                    "total_tokens": total_tokens,
                    "total_time_s": total_time,
                    "cur_gpu_mem_mb": cur_mem,
                    "peak_gpu_mem_mb": peak_mem,
                    "gpu_util_percent": gpu_util,
                    "balance": balance,
                }}
            )
        
        # ---------------------------
        # PROFILER EXAMPLE (mirrors single_gpu)
        # ---------------------------
        steps = 8
        # choose activities dynamically (CPU-only path avoids CUDA activity)
        activities = [ProfilerActivity.CPU]
        if device.type.startswith("cuda"):
            activities.append(ProfilerActivity.CUDA)

        if rank == 0 and it is None:
            # safety, though rank 0 always has a loader
            it = iter(DataLoader(dataset, batch_size=conf["batch_size"], shuffle=True, num_workers=2, pin_memory=True))

        with profile(
            activities=activities,
            profile_memory=True,
            record_shapes=True,
            with_stack=False,
        ) as prof:
            for i in range(steps):
                # Rank 0 fetches a batch
                if rank == 0:
                    try:
                        batch = next(it)
                    except StopIteration:
                        it = iter(DataLoader(dataset, batch_size=conf["batch_size"], shuffle=True, num_workers=2, pin_memory=True))
                        batch = next(it)
                    batch = batch.to(device, non_blocking=True)

                # Broadcast labels to last rank (match training loop)
                if rank == 0:
                    targets_for_last = batch
                    dist.broadcast(targets_for_last, src=0, group=loss_group)
                    targets_last = None
                elif rank == last_rank:
                    targets_last = torch.empty(
                        (conf["batch_size"], conf["seq_len"]), dtype=torch.long, device=device
                    )
                    dist.broadcast(targets_last, src=0, group=loss_group)
                else:
                    targets_last = None

                # Zero grad
                optim.zero_grad(set_to_none=True)

                # Profiled pipeline step
                if rank == 0:
                    with record_function(MODEL_FORWARD_PROFILER_LABEL):
                        schedule.step(batch)
                elif rank == last_rank:
                    # We nest loss/backward labels around the schedule call.
                    # (Internally, GPipe will compute per-microbatch loss then backprop.)
                    with record_function(MODEL_LOSS_PROFILER_LABEL):
                        with record_function(MODEL_BACKWARD_PROFILER_LABEL):
                            schedule.step(target=targets_last, losses=None)
                else:
                    schedule.step()

                # Optimizer step
                with record_function(MODEL_OPTIMIZER_PROFILER_LABEL):
                    optim.step()

        # Log profiler metrics per-rank (simple and effective; DDP-style gather is optional)
        profiler_metrics = {
            "rank": rank,
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
        # Each rank logs its own profile (helps diagnose imbalances per stage)
        logger.info("Profiler metrics (GPipe)", extra={"extra": profiler_metrics})

        # Optional: print a compact table on CPU-only runs like your DDP example
        if device.type == "cpu":
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    finally:
        dist.barrier()
        dist.destroy_process_group()
