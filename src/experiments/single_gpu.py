import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

from models.simple import SimpleTransformerDecoder
from datasets.synthetic import SyntheticDataset
from utils.gpu import (
    compute_throughput,
    gpu_memory_allocated,
    gpu_utilization_percent,
    reset_peak_mem
)


def run_single_gpu_experiment(conf):
    # TODO: This won't work for multi-GPU setups - diff strats have diff wrappers
    # Likely need to pass the model in as an arg - and also the rank and flags indicating types
    model = SimpleTransformerDecoder(
        conf["vocab_size"],
        conf["d_model"],
        conf["n_heads"],
        conf["n_layers"],
        conf["d_ff"],
        conf["seq_len"]
    ).to("cuda")

    print(f"Using configuration: {conf}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    dataset = SyntheticDataset(
        n_samples=10000,
        seq_len=conf["seq_len"],
        vocab_size=conf["vocab_size"]
    )
    # TODO: This won't work for multi-GPU strats, we need to include a "sampler"
    loader = DataLoader(
        dataset,
        batch_size=conf["batch_size"],
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=conf["lr"])
    criterion = nn.CrossEntropyLoss()

    model.train()
    total_tokens = 0
    step = 0
    token_throughputs = []
    sample_throughputs = []
    losses = []
    reset_peak_mem()
    t0 = time.perf_counter()
    warmup = conf["warmup_steps"]
    max_steps = conf["max_steps"]
    it = iter(loader)
    device = "cuda" # TODO: revisit for multi-GPU setup
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
        torch.cuda.synchronize() if device.startswith("cuda") else None
        t_before = time.perf_counter()
        logits = model(batch)

        # shift logits/targets for next-token prediction
        logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))  # (B*(S-1), V)
        targets = batch[:, 1:].contiguous().view(-1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize() if device.startswith("cuda") else None
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
        if step % 10 == 0 or step == max_steps-1:
            print(f"step {step+1}/{max_steps} | "
                  f"loss {loss.item():.4f} | "
                  f"step_time {step_time:.4f}s | "
                  f"tokens/s {tokens/step_time:,.0f} | "
                  f"cur_mem {cur_mem:.1f}MB peak {peak_mem:.1f}MB | "
                  f"gpu_util {gpu_util}"
            )

    total_time = time.perf_counter() - t0
    avg_tokens_per_s = sum(token_throughputs) / len(token_throughputs) if token_throughputs else 0
    avg_samples_per_s = sum(sample_throughputs) / len(sample_throughputs) if sample_throughputs else 0
    avg_loss = sum(losses) / len(losses) if losses else None

    print("")
    print("Results:")
    print(f"    avg_tokens_per_s:  {avg_tokens_per_s}")
    print(f"    avg_samples_per_s: {avg_samples_per_s}")
    print(f"    avg_loss:          {avg_loss}")
    print(f"    total_tokens:      {total_tokens}")
    print(f"    total_time_s:      {total_time}")
    print(f"    cur_mem_mb:        {cur_mem}")
    print(f"    peak_mem_mb:       {peak_mem}")
    print(f"    gpu_util_percent:  {gpu_util}")

    # PROFILER EXAMPLE
    # TODO: Modify to support multi-GPU setups
    steps = 8
    opt = torch.optim.AdamW(model.parameters(), lr=conf["lr"])
    it = iter(loader)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True,
                 record_shapes=True,
                 with_stack=False) as prof:
        for i in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch = batch.to(device, non_blocking=True)
            opt.zero_grad()
            # TODO: This captures forward pass - probably want to include loss + backward steps in their own labels
            with record_function("model_forward"):
                logits = model(batch)
            logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            targets = batch[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            opt.step()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
