"""
When doing distirubted tests on RunPod, I've found that this environment variable prevents
stalls due to comms. This explicitly sets the peer-to-peer/NCCL comms to occur over NVLink.

export NCCL_P2P_LEVEL=NVL
"""
import argparse
import os

import torch

from configs import CONF
from experiments import single_gpu, torch_ddp, torch_gpipe, tensor_parallel
from models.simple import SimpleTransformerDecoder
from utils.device import get_device
from utils.logger import get_logger


CWD = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.environ.get("LOG_PATH", os.path.join(CWD, "..", "logs"))

EXPERIMENT_TYPES = {
    "single_gpu": single_gpu.run_single_gpu_experiment,
    "torch_ddp": torch_ddp.run_torch_ddp_experiment,
    "torch_gpipe": torch_gpipe.run_torch_gpipe_experiment,
    "tensor_parallel": tensor_parallel.run_tensor_parallel_experiment,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the given experiment type for a given configuration",
    )
    parser.add_argument("experiment_type", choices=list(EXPERIMENT_TYPES.keys()))
    parser.add_argument("model_configuration", choices=list(CONF.keys()))
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    conf = CONF[args.model_configuration]
    model = SimpleTransformerDecoder(
        conf["vocab_size"],
        conf["d_model"],
        conf["n_heads"],
        conf["n_layers"],
        conf["d_ff"],
        conf["seq_len"],
        conf["dtype"]
    )

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.dry_run:
        print(f"Using configuration: {conf}")
        print(f"Trainable parameters: {num_param}")
        exit(0)

    force_cpu = args.model_configuration == "cpu"
    if not torch.cuda.is_available() and not force_cpu:
        print(
            "ERROR: CUDA was not detected and you are not running a `cpu` configuration."
        )
        print(
            "You need to either run the `cpu` configuration or ensure you have CUDA enabled and a GPU available."
        )
        print("Exiting...")
        exit(1)

    device = get_device(force_cpu=force_cpu)
    model.to(device=device, dtype=conf["dtype"])

    logger = get_logger(args.experiment_type, args.model_configuration, LOG_PATH)
    logger.info(
        "Starting experiment",
        extra={
            "extra": {
                "configuration": {k: str(v) for k, v in conf.items()},
                "num_param": num_param,
            }
        },
    )
    experiment = EXPERIMENT_TYPES[args.experiment_type]
    experiment(model, conf, device, logger)
