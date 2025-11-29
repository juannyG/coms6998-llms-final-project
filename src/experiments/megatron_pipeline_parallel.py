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
from utils.logger import get_log_file_parent_dir

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_attention_heads = kwargs.get('num_attention_heads', 0)

    def __getitem__(self, idx):
        tokens = super().__getitem__(idx)
        
        # Create Megatron-expected format
        # See: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/gpt_dataset.py#L645-L661
        # We need to do something slightly different: MT PP expects the attention_mask to include a dimension of size # attention heads
        return {
            'tokens': tokens,
            'attention_mask': torch.tril(torch.ones(self.num_attention_heads, self.seq_len, self.seq_len, dtype=torch.bool)),
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


def run_pipeline_parallel_experiment(_, conf, device, logger):
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
        timeout=datetime.timedelta(seconds=10)
    )
    parallel_state.initialize_model_parallel(pipeline_model_parallel_size=world_size)


    # We use world_size here, because world_size == # of stages
    #n_microbatches = world_size # Worst case configuration
    n_microbatches = world_size * 4 # "optimal" configuration - bumped batch size up to 32 across the board
    micro_batch_size = conf["batch_size"] // n_microbatches

    try:
        model_parallel_cuda_manual_seed(123)
        print(f"Rank {rank}: Set random seed")

        tc = TransformerConfig(
            num_layers=conf["n_layers"],
            hidden_size=conf["d_model"],
            num_attention_heads=conf["n_heads"],
            pipeline_dtype=conf["dtype"],
            pipeline_model_parallel_size=world_size,
        )
        print(f"Rank {rank}: Created TransformerConfig")

        gpt_model = GPTModel(
            config=tc,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=conf["vocab_size"],
            max_sequence_length=conf["seq_len"],
            pre_process=(rank == 0),
            post_process=(rank == world_size - 1)
        )
        gpt_model.to(device=device, dtype=conf["dtype"])

        optimizer = Adam(gpt_model.parameters())

        dataset = MegatronSyntheticDataset(
            n_samples=10000,
            seq_len=conf["seq_len"],
            vocab_size=conf["vocab_size"],
            num_attention_heads=conf["n_heads"]
        )
        loader = DataLoader(
            dataset,
            batch_size=micro_batch_size,
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
        gpu_util_per_step = []
        gpu_mem_per_step = []
        loss = torch.Tensor([0])

        reset_peak_mem()
        t0 = time.perf_counter()
        max_steps = conf["max_steps"]
        for step in range(max_steps):
            optimizer.zero_grad()

            torch.cuda.synchronize() if device.type.startswith("cuda") else None
            t_before = time.perf_counter()

            losses_reduced = forward_back_func(
                forward_step_func=forward_step_func,
                data_iterator=it,
                model=gpt_model,
                num_microbatches=n_microbatches,
                micro_batch_size=micro_batch_size,
                seq_length=conf["seq_len"],
                decoder_seq_length=conf["seq_len"],
                forward_only=False,
            )

            optimizer.step()

            torch.cuda.synchronize() if device.type.startswith("cuda") else None
            t_after = time.perf_counter()

            # metrics
            step_time = t_after - t_before

            # These are constants, but, whatever - do it each time in the loop...
            # Tweaked computation: Tokens per step = batch_size * (seq_len - 1) * num_microbatches
            if rank == world_size - 1:
                total_tokens += micro_batch_size * (conf["seq_len"] - 1) * n_microbatches
                loss = losses_reduced[0]["loss"]

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
        total_throughput = 0
        avg_gpu_mem_mb = 0
        avg_gpu_util_percent = 0
        if rank == world_size - 1:
            # TODO: Explain why we're only using last rank for these metrics
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
                optimizer.zero_grad()

                torch.cuda.synchronize() if device.type.startswith("cuda") else None

                with record_function(MODEL_FORWARD_BACKWARD_PROFILER_LABEL):
                    losses_reduced = forward_back_func(
                        forward_step_func=forward_step_func,
                        data_iterator=it,
                        model=gpt_model,
                        num_microbatches=n_microbatches,
                        micro_batch_size=micro_batch_size,
                        seq_length=conf["seq_len"],
                        decoder_seq_length=conf["seq_len"],
                        forward_only=False,
                    )

                with record_function(MODEL_OPTIMIZER_PROFILER_LABEL):
                    optimizer.step()

                torch.cuda.synchronize() if device.type.startswith("cuda") else None

                prof.step()
    except Exception as exc:
        print(exc)
        raise exc
    finally:
        torch.distributed.destroy_process_group()
