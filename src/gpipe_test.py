import torch
import torch.nn as nn

from configs import CONF
from torch.optim.optimizer import _to_scalar
from utils.device import get_device

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
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
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
        return torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).to(device)

device = get_device()
conf = CONF['cpu']
model = SimpleTransformerDecoder(
    conf["vocab_size"],
    conf["d_model"],
    2,
    6, # Note: --nproc_per_node=N must have N >= n_layers
    conf["d_ff"],
    conf["seq_len"],
)

##### Startin gpipe stuff

# See: huggingface GPT2 gpipe example: https://github.com/pytorch/PiPPy/blob/main/examples/huggingface/pippy_gpt2.py
import os
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, ScheduleGPipe, SplitPoint

# Initialize distributed training
if torch.cuda.is_available():
    dist.init_process_group(backend='nccl')  # or 'gloo' for CPU
else:
    dist.init_process_group(backend='gloo')  # or 'gloo' for CPU


# Get rank and world size
world_size = dist.get_world_size()
rank = dist.get_rank()

if torch.cuda.is_available():
    torch.cuda.set_device(rank)

if world_size == 1:
    print("Even on CPU, we need --nporc_per_node > 1")
    exit(1)

if rank == 0:
    print(conf)
    print(model)

decoders_per_rank = (conf["n_layers"] + world_size - 1) // world_size
print(f"[{rank}]: decoders_per_rank = {decoders_per_rank}")
split_spec = {
    f'layers.{i * decoders_per_rank}': SplitPoint.BEGINNING
    for i in range(1, world_size)
}

# TODO: we don't use explicitly use conf["batch_size"], we should actually use batch_size / n_microbatches
# Implied - make sure n_microbatches is divisible evenly by batch_size....ideally, n_microbatches = 2 * world_size
conf["batch_size"] = 32
n_microbatches = 4
x = torch.randint(0, conf["vocab_size"], (conf["batch_size"] // n_microbatches, conf["seq_len"])) 
pipe = pipeline(
    module=model,
    mb_args=(x,),
    split_spec=split_spec,
)
print(f"[{rank}]", pipe)

stage = pipe.build_stage(rank, device)
print(f"[{rank}] {stage.submod}")

# Create GPipe scheduler
# chunks parameter determines how many microbatches to split the batch into
# SIMILAR TO ABOVE NOTE: Note: --nproc_per_node=N must have N >= n_layers
# Number of microbatches must be >= number of stages
schedule = ScheduleGPipe(stage, n_microbatches)

# Create optimizer for the pipeline stage
optimizer = torch.optim.AdamW(stage.submod.parameters(), lr=1e-4)
# Calculate loss (cross-entropy for language modeling)
loss_fn = nn.CrossEntropyLoss()

# Training loop
num_steps = 100
for step in range(num_steps):
    optimizer.zero_grad()
    
    # Prepare input and target
    # For language modeling, target is typically input shifted by 1
    inputs = torch.randint(0, conf["vocab_size"], (conf["batch_size"], conf["seq_len"]))
    targets = torch.randint(0, conf["vocab_size"], (conf["batch_size"], conf["seq_len"]))
    
    # Forward and backward pass through pipeline
    if rank == 0:
        # First stage sends input
        schedule.step(inputs.to(device))
    elif rank == world_size - 1:
        # Last stage receives output and calculates loss
        output = schedule.step()
        
        # Reshape for loss calculation: (batch_size * seq_len, vocab_size) and (batch_size * seq_len,)
        loss = loss_fn(output.view(-1, conf["vocab_size"]), targets.view(-1).to(device))
        
        # Backward pass starts from last stage
        loss.backward()
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
    else:
        # Intermediate stages just pass data through
        schedule.step()
    
    # Update parameters
    optimizer.step()

# Cleanup
dist.destroy_process_group()
