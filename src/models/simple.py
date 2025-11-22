import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        attn_ln = self.ln1(x)
        attn_out, _ = self.attn(attn_ln, attn_ln, attn_ln, attn_mask=attn_mask)
        x = x + attn_out
        ff_ln = self.ln2(x)
        return x + self.ff(ff_ln)


class SimpleTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.drop = nn.Dropout(0.1)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        B, S = tokens.shape
        x = self.tok_emb(tokens) + self.pos_emb[:, :S, :]
        x = self.drop(x)
        mask = self._causal_mask(S, x.device)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)
        return self.head(x)

    @staticmethod
    def _causal_mask(seq_len, device):
        return torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).to(
            device
        )
