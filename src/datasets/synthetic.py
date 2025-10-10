import random

import torch
from torch.utils.data import Dataset


class SyntheticLM(Dataset):
    def __init__(self, n_samples, seq_len, vocab_size, seed=1111):
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
