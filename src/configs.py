import torch


CONF = {
    "baseline": {
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
