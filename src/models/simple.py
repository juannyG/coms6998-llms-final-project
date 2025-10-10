import torch
import torch.nn as nn


class SimpleTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.drop = nn.Dropout(0.1)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "ln1": nn.LayerNorm(d_model),
                "ff": nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model),
                ),
                "ln2": nn.LayerNorm(d_model)
            }) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        B, S = tokens.shape
        x = self.tok_emb(tokens) + self.pos_emb[:, :S, :]
        x = self.drop(x)
        for layer in self.layers:
            attn_ln = layer["ln1"](x)
            attn_out, _ = layer["attn"](attn_ln, attn_ln, attn_ln, attn_mask=self._causal_mask(S, x.device))
            x = x + attn_out
            ff_ln = layer["ln2"](x)
            x = x + layer["ff"](ff_ln)
        x = self.ln_f(x)  # (B, S, D)
        logits = self.head(x)  # (B, S, V)
        return logits

    @staticmethod
    def _causal_mask(seq_len, device):
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).to(device)
        return mask
