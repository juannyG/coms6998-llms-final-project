#!/bin/sh
#SBATCH --account=edu
#SBATCH --job-name=zero
#SBATCH --gres=gpu:2
#SBATCH -c 1
#SBATCH --time=0-01:00
#SBATCH --mem-per-cpu=5gb

module load cuda
source ../../bin/activate

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=30

# pick which ZeRO config to use:
export ZERO_CONFIG=src/zero_configs/zero_2g_stage3.yaml
# export ZERO_CONFIG=src/zero_configs/zero_2g_stage1.yaml
# export ZERO_CONFIG=src/zero_configs/zero_2g_stage2.yaml
# export ZERO_CONFIG=src/zero_configs/zero_2g_stage3_offload.yaml

torchrun --standalone --nproc_per_node=2 run_experiment.py zero 10m

