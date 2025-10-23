import math

import torch
import torch.nn as nn
from megatron.core import tensor_parallel
from megatron.core.model_parallel_config import ModelParallelConfig


def pytorch_linear_init(weights):
    """
    MGT requires you to handle weight initialization, where as PyTorch provides some
    kind of "reasonable defaults."

    Attempting to keep "experiment parity", we try to initialize weights the same way
    PyTorch does, see: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L108-L128
    """
    nn.init.kaiming_uniform_(weights, a=math.sqrt(5))


class SimpleMegatronTransformerDecoder(nn.Module):
    """
    Reconstruction of SimpleTransformerDecoder, but using Megatron based components.
    This helps us test Megatron's effects on training, while keeping experimental
    consistency: using Megatron's GPT could produce significantly different results
    simply due to how different its construction is to our SimpleTransformerDecoder.
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, seq_len):
        super().__init__()

        # TODO: Depending on how this plays with initialize_megatron(), we may need to move this/pass it in
        # All things megatron require a configuration
        mgt_parallel_config = ModelParallelConfig()

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.drop = nn.Dropout(0.1)
        self.layers = nn.ModuleList([]) # TODO: 

        # We need to map from the model dimension back to the vocab dimension
        # The MGT ouptut layer is a column parallel op: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/models/gpt/gpt_model.py#L234C49-L234C69
        # TODO: Run an experiment with Row vs Column parallel ops and see if it has any effect
        self.head = tensor_parallel.ColumnParallelLinear(
            d_model,
            vocab_size,
            config=mgt_parallel_config,
            init_method=pytorch_linear_init,
            bias=False,
            skip_bias_add=False,
        )
        # TODO: Reminder: self.head(x) can still be used, but in megatron land, it returns a tuple: logits, bias
        # We need to be aware of this in `def forward` otherwise we'll be passing garbage around
