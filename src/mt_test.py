import os

import torch
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

import configs


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    parallel_state.destroy_model_parallel()
    torch.distributed.init_process_group(backend="gloo")
    parallel_state.initialize_model_parallel(1, 1)

    model_parallel_cuda_manual_seed(123)

    conf = configs.CONF["10m"]
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

    print(gpt_model)
    print(f"Total parameters: {sum(p.numel() for p in gpt_model.parameters())}")
