import argparse

import torch

from configs import CONF
from experiments.single_gpu import run_single_gpu_experiment

EXPERIMENT_TYPES = {
    'single_gpu': run_single_gpu_experiment,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run the given experiment type for a given configuration",
    )
    parser.add_argument('experiment_type', choices=list(EXPERIMENT_TYPES.keys()))
    parser.add_argument('configuration', choices=list(CONF.keys()))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA was not detected")
        print("These experiments are meant to be run in an environment with CUDA and at least one GPU.")
        print("Exiting...")
        exit(1)

    experiment = EXPERIMENT_TYPES[args.experiment_type]
    experiment(CONF[args.configuration])
