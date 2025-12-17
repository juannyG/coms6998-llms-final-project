# Distributed Training Strategies on Commodity Hardware (PCIe)

Authors:
* Can Kerem Akbulut, cka2115@columbia.edu
* Rakene Chowdhury, rc3574@columbia.edu
* Juan Gutierrez, jmg2048@columbia.edu

Paper: [See here](overleaf/Fall25-COMS6998-Scaling_LLMs-Distributed_Training_Strategies_on_Commodity_Hardware.pdf)

W&B Report: [See here](https://wandb.ai/jmg2048-columbia-university/fall25-sllm-final-project/reports/Distributed-Training-Strategies-on-Commodity-Hardware-Dashboard--VmlldzoxNTM4MTA1Mg?accessToken=npt0nk8y29m84ueagcb8t8p69j2vz7nb596arm0pa1r54nvgs7zdr1d7741f16rf)

This repository contains the code used to run and aggregate experiments comparing distributed training strategies (Megatron-LM TP/DDP/PP and DeepSpeed ZeRO) on PCIe-connected NVIDIA RTX A6000 GPUs. Experiments measure throughput, scaling efficiency, and GPU memory usage across models from 10M to 1B parameters. Experiments use a fixed-shape synthetic dataset to eliminate data variability and do not measure model quality.

## Repository layout
* `src/run_experiment.py`: main driver invoked by Make targets
* `src/experiments/megatron_ddp.py`, `src/experiments/tensor_parallel.py`, `src/experiments/megatron_pipeline_parallel.py`: Megatron-LM experiments
* `src/experiments/simple_zero.py` + `src/experiments/simple_single.py`: GPT-like PyTorch model used for ZeRO tests
* `src/configs.py`: model configs (10M through 1B) and key hyperparameters
* `src/datasets/synthetic.py`: deterministic synthetic dataset
* `src/tools/metrics/*`: aggregation scripts which compute/format summary tables + CSVs
* `results/`: raw, aggregated, and plotted data from our experimental runs

## Environment Setup

Hardware Requirements:
* NVIDIA GPUs (tested on RTX A6000)
* CUDA + NCCL support
* Multi-GPU system for distributed experiments

Note: Experiments were run on PCIe-connected GPUs (no NVLink).

We recommend using a Python virtual environment, specifically Python 3.12.3.

```bash
python -m venv venv
source venv/bin/activate

git clone git@github.com:juannyG/coms6998-llms-final-project.git proj
cd proj

pip install -r requirements.txt

cd src
```

## Running Experiments

All experiments are orchestrated via a [Makefile](src/Makefile). Logs are written to a configurable directory and later aggregated into tables and CSVs.

Before running experiments, create a log directory:

```bash
mkdir -p ../logs
```

To run the full suite of experiments:

```bash
make all
```
The Makefile defines targets by:
- Strategy (megatron_ddp, megatron_tensor, megatron_pipeline, zero)
- Model size (10M, 100M, 300M, 500M, 1B)
- Number of GPUs (1, 2, 4)

See the Makefile for the complete list of targets and exact naming.


