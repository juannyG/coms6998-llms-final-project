import os
import time
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
from torch.utils.data import DataLoader

from datasets.synthetic import SyntheticDataset
from utils.device import get_device
from utils.gpu import (
    gpu_memory_allocated,
    gpu_utilization_percent,
    reset_peak_mem,
)


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
    torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)
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
        gpt_model.to(device)

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

        warmup = conf["warmup_steps"]
        max_steps = conf["max_steps"]
        for step in range(max_steps):
            optimizer.zero_grad()
            
            losses_reduced = forward_back_func(
                forward_step_func=forward_step_func,
                data_iterator=it,
                model=gpt_model,
                num_microbatches=1,
                seq_length=conf["seq_len"],
                micro_batch_size=8,
                decoder_seq_length=conf["seq_len"],
                forward_only=False,
            )
            # NOTE: We're not using this here so we can measure pure TP. This would be used if were doing DDP/pipeline as well...
            # finalize_model_grads([gpt_model])

            optimizer.step()

            if step % 10 == 0 or step == max_steps - 1:
                print(f"Iteration {step}: Losses reduced: {losses_reduced}")
    finally:
        torch.distributed.destroy_process_group()
