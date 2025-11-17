"""
Our model configurations are inspired by established architectural scaling trends observed in GPT-2, BERT,
and subsequent work (Radford et al., 2019; Kaplan et al., 2020; Hoffmann et al., 2022).

Using GPT-2 / BERT `d_model` values, we then compute
* n_heads =~ d_model // 64
* n_layers =~ sqrt(param count / 100M )
* d_dff =~ 4 * d_model

Learning rate, `lr`, follows the standard 2e-4 and batch size is modified relative to the model size.

`d_model` value references can be found:
* Radford et al., (2019); Table 2  - https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
* Devlin et al., (2018); Section 3 - https://arxiv.org/pdf/1810.04805
"""

import torch


CONF = {
    # Use this configuration if there's HPC congestion and you want feedback quickly
    # NOTE: Megatron does not work at all with CPU
    "cpu": {
        "vocab_size": 800,
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 4,
        "d_ff": 128,
        "seq_len": 32,
        "batch_size": 8,
        "lr": 0.0002,
        "dtype": torch.float32,
        "max_steps": 200,
        "warmup_steps": 10,
    },
    "10m": {
        "vocab_size": 8000,
        "d_model": 320,
        "n_heads": 8,
        "n_layers": 4,
        "d_ff": 1536,
        "seq_len": 128,
        "batch_size": 16,
        "lr": 0.0002,
        "dtype": torch.float32,
        "max_steps": 200,
        "warmup_steps": 20,
    },
    "100m": {
        "vocab_size": 8000,
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
        "d_ff": 4 * 768,
        "seq_len": 512,
        "batch_size": 16,
        "lr": 2e-4,
        "dtype": torch.bfloat16,
        "max_steps": 200,
        "warmup_steps": 20,
    },
    "300m": {
        "vocab_size": 8000,
        "d_model": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "d_ff": 4 * 1024,
        "seq_len": 1024,
        "batch_size": 8,  # bring this down because we're doing more... we can boost it once we see how it behaves
        "lr": 2e-4,
        "dtype": torch.bfloat16,
        "max_steps": 200,
        "warmup_steps": 20,
    },
    "500m": {
        "vocab_size": 8000,
        "d_model": 1280,
        "n_heads": 20,
        "n_layers": 24,
        "d_ff": 4 * 1280,
        "seq_len": 1024,
        "batch_size": 8,
        "lr": 2e-4,
        "dtype": torch.bfloat16,
        "max_steps": 200,
        "warmup_steps": 20,
    },
    "1b": {
        "vocab_size": 8000,
        "d_model": 1536,
        "n_heads": 24,
        "n_layers": 36,
        "d_ff": 4 * 1536,
        "seq_len": 1024,
        "batch_size": 4,  # bring it down even further...
        "lr": 2e-4,
        "dtype": torch.bfloat16,
        "max_steps": 200,
        "warmup_steps": 20,
    },
}
