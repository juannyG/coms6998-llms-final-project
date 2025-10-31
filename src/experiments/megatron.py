import os

import torch
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig


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
    rank: int = int(os.environ["LOCAL_RANK"])
    world_size: int = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)

    try:
        model_parallel_cuda_manual_seed(123)

        tc = TransformerConfig(
            num_layers=conf["n_layers"],
            hidden_size=conf["d_model"],
            num_attention_heads=conf["n_heads"],
            use_cpu_initialization=True,     # avoid GPU init kernels
        )

        gpt_model = GPTModel(
            config=tc,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=conf["vocab_size"],
            max_sequence_length=conf["seq_len"],
        )

        # TODO: Remove these...
        print(gpt_model)
        print(f"Total parameters: {sum(p.numel() for p in gpt_model.parameters())}")
    finally:
        torch.distributed.destroy_process_group()

