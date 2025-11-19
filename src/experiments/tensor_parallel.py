import os
import time
import datetime
from functools import partial

import torch
from torch.optim import Adam
from megatron.core import parallel_state
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader

from datasets.synthetic import SyntheticDataset
from tools.metrics.metrics_dataclasses import TrainingResults
from utils.device import get_device
from utils.gpu import (
    gpu_memory_allocated,
    gpu_utilization_percent,
    reset_peak_mem,
)

MODEL_FORWARD_PROFILER_LABEL = "model_forward"
MODEL_LOSS_PROFILER_LABEL = "model_loss"
MODEL_FORWARD_BACKWARD_PROFILER_LABEL = "model_forward_backward"
MODEL_OPTIMIZER_PROFILER_LABEL = "model_optimizer_step"
EXPERIMENT_PROFILER_LABELS = [
    MODEL_FORWARD_PROFILER_LABEL,
    MODEL_LOSS_PROFILER_LABEL,
    MODEL_FORWARD_BACKWARD_PROFILER_LABEL,
    MODEL_OPTIMIZER_PROFILER_LABEL,
]


class MegatronSyntheticDataset(SyntheticDataset):
    def __getitem__(self, idx):
        tokens = super().__getitem__(idx)
        
        # Create Megatron-expected format
        # See: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/gpt_dataset.py#L645-L661
        return {
            'tokens': tokens,
            'attention_mask': torch.tril(torch.ones((self.seq_len, self.seq_len), dtype=torch.bool)).unsqueeze(0),
            'position_ids': torch.arange(self.seq_len, dtype=torch.long),
            'labels': tokens.clone(), # We need something...so just use the tokens...
            'loss_mask': torch.ones(self.seq_len, dtype=torch.float)
        }


def forward_step_func(data_iterator, gpt_model):
    """
    SEE: https://github.com/NVIDIA/Megatron-LM/blob/main/examples/run_simple_mcore_train_loop.py#L114

    Forward step function that computes model output and returns loss function.

    Args:
        data_iterator: Iterator providing training batches.
        model: The GPT model to train.

    Returns:
        Tuple of (output_tensor, loss_function) where loss_function is a partial
        function that will compute the final loss when called.
    """

    def loss_func(loss_mask, output_tensor):
        with record_function(MODEL_LOSS_PROFILER_LABEL):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
            # If you have data parallel reduce loss across data parallel groups.
            # If pipeline parallel, loss computation is done only in last stage.

            return loss, {"loss": loss}

    device = get_device()
    data = next(data_iterator)
    tokens = data["tokens"].to(device)
    attention_mask = data["attention_mask"].to(device)
    position_ids = data["position_ids"].to(device)
    labels = data["labels"].to(device)
    loss_mask = data["loss_mask"].to(device)

    with record_function(MODEL_FORWARD_PROFILER_LABEL):
        output_tensor = gpt_model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def run_tensor_parallel_experiment(_, conf, device, logger):
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
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(seconds=30)
    )
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)

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

        # TODO: Remove these...
        #print(gpt_model)
        #print(f"Total parameters: {sum(p.numel() for p in gpt_model.parameters())}")

        optimizer = Adam(gpt_model.parameters())

        dataset = MegatronSyntheticDataset(
            n_samples=10000, 
            seq_len=conf["seq_len"], 
            vocab_size=conf["vocab_size"],
        )
        loader = DataLoader(
            dataset,
            batch_size=conf["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        it = iter(loader)

        forward_back_func = get_forward_backward_func()

        gpt_model.train()
        step = 0
        total_tokens = 0
        cur_mem = 0
        peak_mem = 0
        gpu_util = 0
        token_throughputs = []
        sample_throughputs = []
        gpu_util_per_step = []
        gpu_mem_per_step = []
        losses = []
        reset_peak_mem()
        t0 = time.perf_counter()
        warmup = conf["warmup_steps"]
        max_steps = conf["max_steps"]
        for step in range(max_steps):
            optimizer.zero_grad()
            
            torch.cuda.synchronize() if device.type.startswith("cuda") else None
            t_before = time.perf_counter()

            losses_reduced = forward_back_func(
                forward_step_func=forward_step_func,
                data_iterator=it,
                model=gpt_model,
                num_microbatches=1,
                seq_length=conf["seq_len"],
                micro_batch_size=conf["batch_size"],
                decoder_seq_length=conf["seq_len"],
                forward_only=False,
            )
            # NOTE: We're not using this here so we can measure pure TP. This would be used if were doing DDP/pipeline as well...
            # finalize_model_grads([gpt_model])

            optimizer.step()

            torch.cuda.synchronize() if device.type.startswith("cuda") else None
            t_after = time.perf_counter()

            # metrics
            step_time = t_after - t_before
            # We have to do things in a slightly different way, given that the dimensions are buried in the `forward_step_func` closure, but we can still compute tokens based on our parameters...
            #tokens = B * (S - 1)  # tokens processed for training step
            #samples = B
            #total_tokens += tokens
            micro_batch_size = conf["batch_size"]
            seq_length = conf["seq_len"]
            num_microbatches = 1

            # These are constants, but, whatever - do it each time in the loop...
            # Tweaked computation: Tokens per step = batch_size * (seq_len - 1) * num_microbatches
            tokens = micro_batch_size * (seq_length - 1) * num_microbatches
            samples = micro_batch_size * num_microbatches
            total_tokens += tokens
            if step >= warmup:
                token_throughputs.append(tokens / step_time)
                sample_throughputs.append(samples / step_time)
                # TODO: Is it losses_reduced[0] because we're only doing single GPU stuff? Need to check this..
                losses.append(losses_reduced[0]['loss'].item())

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
                            "loss": f"{losses_reduced[0]['loss'].item():.4f}",
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
            sum(sample_throughputs) / len(sample_throughputs)
            if sample_throughputs
            else 0
        )
        avg_loss = sum(losses) / len(losses) if losses else None

        avg_gpu_mem_mb = (
            sum(gpu_mem_per_step) / len(gpu_mem_per_step) if gpu_mem_per_step else 0
        )
        avg_gpu_util_percent = (
            sum(gpu_util_per_step) / len(gpu_util_per_step) if gpu_util_per_step else 0
        )

        training_results = TrainingResults(
            avg_tokens_per_s=avg_tokens_per_s,
            avg_samples_per_s=avg_samples_per_s,
            avg_loss=avg_loss,
            total_tokens=total_tokens,
            total_time_s=total_time,
            avg_gpu_mem_mb=avg_gpu_mem_mb,
            peak_gpu_mem_mb=peak_mem,
            avg_gpu_util_percent=avg_gpu_util_percent,
        )
        logger.info(
            "Training results",
            extra={"extra": training_results.to_dict()},
        )

        steps = 8
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            with_stack=False,
        ) as prof:
            for i in range(steps):
                optimizer.zero_grad()

                with record_function(MODEL_FORWARD_BACKWARD_PROFILER_LABEL):
                    losses_reduced = forward_back_func(
                        forward_step_func=forward_step_func,
                        data_iterator=it,
                        model=gpt_model,
                        num_microbatches=1,
                        seq_length=conf["seq_len"],
                        micro_batch_size=conf["batch_size"],
                        decoder_seq_length=conf["seq_len"],
                        forward_only=False,
                    )

                with record_function(MODEL_OPTIMIZER_PROFILER_LABEL):
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
    finally:
        torch.distributed.destroy_process_group()
