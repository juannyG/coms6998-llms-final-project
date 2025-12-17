# Distributed Training Strategies on Commodity Hardware (PCIe)

Authors:
* Can Kerem Akbulut, cka2115@columbia.edu
* Rakene Chowdhury, rc3574@columbia.edu
* Juan Gutierrez, jmg2048@columbia.edu

Paper: [See here](overleaf/Fall25-COMS6998-Scaling_LLMs-Distributed_Training_Strategies_on_Commodity_Hardware.pdf)

W&B Report: [See here](https://wandb.ai/jmg2048-columbia-university/fall25-sllm-final-project/reports/Distributed-Training-Strategies-on-Commodity-Hardware-Dashboard--VmlldzoxNTM4MTA1Mg?accessToken=npt0nk8y29m84ueagcb8t8p69j2vz7nb596arm0pa1r54nvgs7zdr1d7741f16rf)

This repository contains the code used to run and aggregate experiments comparing distributed training strategies (Megatron-LM TP/DDP/PP and DeepSpeed ZeRO) on PCIe-connected NVIDIA RTX A6000 GPUs. Experiments measure throughput, scaling efficiency, and GPU memory usage across models from 10M to 1B parameters. Experiments use a fixed-shape synthetic dataset to eliminate data variability and do not measure model quality.

## Repository layout

```
.
├── results                                   # Raw, aggregated, and plotted data from our experimental runs
└── src
    ├── configs.py                            # Model configurations (10M through 1B) and hyperparameters
    ├── datasets
    │   └── synthetic.py                      # Deterministic fixed-shape synthetic dataset
    ├── experiments
    │   ├── megatron_ddp.py                   # Megatron data parallelism experiment
    │   ├── megatron_pipeline_parallel.py     # Megatron pipeline parallelism experiment
    │   ├── tensor_parallel.py                # Megatron tensor parallelism experiment
    │   ├── simple_single.py                  # Single GPU baseline (PyTorch GPT-like model)
    │   ├── simple_zero.py                    # ZeRO experiments using GPT-like PyTorch model
    ├── run_experiment.py                     # Main driver invoked by Make targets
    ├── tools
    │   ├── metrics                           # Metric aggregation scripts (summary tables + CSVs)
    ├── zero_configs                          # ZeRO configuration YAMLs
    ├── Makefile                              # Experiment orchestration
└── README.md
```

## Environment Setup

Hardware Requirements:
* NVIDIA GPUs (tested on RTX A6000)
* CUDA + NCCL support
* Multi-GPU system for distributed experiments

Note: Experiments were run on PCIe-connected GPUs (no NVLink).

We recommend using a Python virtual environment, tested with Python 3.12.3.

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

Note: Re-running the full experiment suite requires a multi-GPU system and can take several hours. The repository includes raw results and aggregated CSVs used in the paper.

Before running experiments, create a log directory:

```bash
mkdir -p ../logs
```

To run the full suite of experiments:

```bash
make all
```
The Makefile defines targets by:
* Strategy (megatron_ddp, megatron_tensor, megatron_pipeline, zero)
* Model size (10M, 100M, 300M, 500M, 1B)
* Number of GPUs (1, 2, 4)

See the [Makefile](src/Makefile) for the complete list of targets and exact naming.

## Metrics and Result Aggregation

After experiments complete, logs are aggregated into:
* CLI-formatted tables
* CSV files used for plotting and analysis

See the [Makefile](src/Makefile) for generating CLI-formatted tables, but here's one example to compare the 1B ZeRO stage3-offload runs on 2 and 4 GPUs against the 1B single GPU baseline:
```bash
make metrics_zero3offload_1b
```

The metrics script(s) assume files are located in `../logs`. If you wish to recreate with our results, you'll need to create a softlink to the specific result types:
```
ln -s results/raw_data/20251128-simple-single-and-zero-logs ../logs
```

Now you can use the [Makefile](src/Makefile) to view any CLI-formatted comparison tables for any of our ZeRO runs.

Plot and CSV generation scripts assume specific log directory layouts. The plot generators contain hardcoded paths. These reflect the structure used during the experiments and were not fully generalized due to time constraints.

Our Megatron and ZeRO runs were located in separate log paths, e.g. `../logs/megatron/*` and `../logs/zero/*` which we manually created and moved the raw data to.

This allowed us to generate our specific CSVs, e.g. for Megatron:
```
python tools/metrics/summary.py all --dir ../logs/megatron/ --target-file ../results/single-tp-dpp-pp-experiment-results.csv
```

The plot generators hardcode the `../results/*.csv` seen in the repository.

## Notes and Limitations

- ZeRO experiments use a lightweight GPT-like PyTorch model due to incompatibility between Megatron's GPTModel and standalone ZeRO.
- ZeRO results are analyzed independently and not directly compared against Megatron results.
- Some scripts assume specific directory layouts and were not fully refactored for general use.
