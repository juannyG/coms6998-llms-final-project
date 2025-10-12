import argparse

import torch

from configs import CONF
from experiments.single_gpu import run_single_gpu_experiment
from models.simple import SimpleTransformerDecoder

EXPERIMENT_TYPES = {
    'single_gpu': run_single_gpu_experiment,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run the given experiment type for a given configuration",
    )
    parser.add_argument('experiment_type', choices=list(EXPERIMENT_TYPES.keys()))
    parser.add_argument('model_configuration', choices=list(CONF.keys()))
    parser.add_argument('--dry-run', action='store_true', default=False)
    args = parser.parse_args()

    device = "cuda"
    if args.model_configuration == "cpu":
        device = "cpu"

    if not torch.cuda.is_available() and device != "cpu":
        print("ERROR: CUDA was not detected and you are not running a `cpu` configuration.")
        print("You need to either run the `cpu` configuration or ensure you have CUDA enabled and a GPU available.")
        print("Exiting...")
        exit(1)

    conf = CONF[args.model_configuration]
    model = SimpleTransformerDecoder(
        conf["vocab_size"],
        conf["d_model"],
        conf["n_heads"],
        conf["n_layers"],
        conf["d_ff"],
        conf["seq_len"]
    ).to(device)

    print(f"Using configuration: {conf}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if not args.dry_run:
        experiment = EXPERIMENT_TYPES[args.experiment_type]
        experiment(model, conf, device)
