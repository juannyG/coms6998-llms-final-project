import torch


CONF = {
    # Use this configuration if there's HPC congestion and you want feedback quickly
    # NOTE: Megatron does not work at all with CPU
    "cpu": {
        "vocab_size": 800,
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 128,
        "seq_len": 32,
        "batch_size": 8,
        "lr": 0.0002,
        "dtype": torch.float32,
        "max_steps": 200,
        "warmup_steps": 10
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
        "warmup_steps": 10
    }
}
