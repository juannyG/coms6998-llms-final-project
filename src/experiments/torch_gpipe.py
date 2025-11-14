"""
Citations:
* https://docs.pytorch.org/docs/2.9/distributed.pipelining.html#module-torch.distributed.pipelining.stage
* https://github.com/pytorch/PiPPy/blob/main/examples/huggingface/pippy_gpt2.py

GPipe experiment (PyTorch built-in, alpha)

Usage:
  torchrun --standalone --nproc_per_node=<NUM_STAGES> run_experiment.py torch_gpipe <CONF_KEY>

Notes:
- One process per pipeline stage.
- Rank 0 owns the input loader; the last rank computes the next-token loss.
- We broadcast the token batch from rank 0 to the last rank each step (only those two ranks participate).
"""

import datetime
import time

import torch
import torch.distributed as dist

from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe
from torch.utils.data import DataLoader
# from torch.profiler import profile, record_function, ProfilerActivity

from datasets.synthetic import SyntheticDataset
from utils.gpu import (
    gpu_memory_allocated,
    gpu_utilization_percent,
    reset_peak_mem,
)

EXPERIMENT_PROFILER_LABELS = []


def run_torch_gpipe_experiment(model, conf, device, logger):
    if torch.cuda.is_available():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=180))
    else:
        dist.init_process_group(backend="gloo", timeout=datetime.timedelta(seconds=180))

    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        model.to(device)

        if world_size == 1:
            raise Exception(
                f"Even on CPU, --nproc_per_node must be > 1: given {world_size}"
            )

        if conf["n_layers"] < world_size:
            raise Exception(
                f"Model layers must be >= world_size: {conf['n_layers']} is less than {world_size}"
            )

        # TODO: make it configurable, but for now, use ideal n_microbatches = 2 * world_size
        # NOTE: we use world_size as stage count, but more sophisticated setups can mave multiple stages per device
        n_microbatches = 2 * world_size
        if n_microbatches < world_size:
            raise Exception(
                f"Number of microbatches must be >= stage count aka world_size: {n_microbatches} is less than {world_size}"
            )

        """
        Build the "split_spec" for pytorch's automatic models splitter.
        This tells `pipeline` how to build the new model graph, which also
        makes it extremely easy to schedule the stage in the pipeline.

        This split spec evenly distributes the layers across all the devices we have.
        An example helps me keep this straight:

        With world_size=3, n_layers=12, we get decoders_per_rank=4:
        i=1: 'layers.4': SplitPoint.BEGINNING  # Start GPU 1 at layer 4
        i=2: 'layers.8': SplitPoint.BEGINNING  # Start GPU 2 at layer 8

        GPU 0 gets: layers 0, 3 (everything before first split)
        GPU 1 gets: layers 4, 7 (from layers.4 to next split)  
        GPU 2 gets: layers 8, 11 (from layers.8 to end)
        """
        decoders_per_rank = (conf["n_layers"] + world_size - 1) // world_size
        print(f"[{rank}]: decoders_per_rank = {decoders_per_rank}")
        split_spec = {
            f'layers.{i * decoders_per_rank}': SplitPoint.BEGINNING
            for i in range(1, world_size)
        }

        # NOTE: Because gpipe uses microbatches of the batch_size, the tensor given to the forward function is batch_size // n_microbatches
        # We also need a "starter" tensor for the pipeline to understand what the "type" of arguments it will receive.
        # The `pipeline` method is 
        gpipe_batch_size = conf["batch_size"] // n_microbatches
        init_tensor = torch.randint(0, conf["vocab_size"], (gpipe_batch_size, conf["seq_len"])) 
        pipe = pipeline(
            module=model,
            mb_args=(init_tensor,),
            split_spec=split_spec,
        )
        #print(f"[{rank}]", pipe)
        
        """
        For simplicity, each device is a rank. Gpipe can support more sophosticated staging, but
        we're not getting into that...not yet at least.
        """
        stage = pipe.build_stage(rank, device)

        dataset = SyntheticDataset(
            n_samples=10000, seq_len=conf["seq_len"], vocab_size=conf["vocab_size"]
        )
        if rank == 0:
            pass
    finally:
        dist.barrier()
        dist.destroy_process_group()
