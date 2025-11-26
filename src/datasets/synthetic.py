import random

import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, n_samples, seq_len, vocab_size, seed=1111, *args, **kwargs):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.rng = random.Random(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        base = (idx * 1009) % 1000003 # use big primes for deterministic "hashing"; more entropy than just `idx`
        self.rng.seed(base)
        tokens = [self.rng.randrange(self.vocab_size) for _ in range(self.seq_len)]
        tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens


class MegatronSyntheticDataset(SyntheticDataset):
    def __getitem__(self, idx):
        tokens = super().__getitem__(idx)

        # Create Megatron-expected format
        # See: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/gpt_dataset.py#L645-L661
        return {
            "tokens": tokens,
            "attention_mask": torch.tril(
                torch.ones((self.seq_len, self.seq_len), dtype=torch.bool)
            ).unsqueeze(0),
            "position_ids": torch.arange(self.seq_len, dtype=torch.long),
            "labels": tokens.clone(),  # We need something...so just use the tokens...
            "loss_mask": torch.ones(self.seq_len, dtype=torch.float),
        }
