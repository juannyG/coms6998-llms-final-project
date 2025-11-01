#!/bin/sh
#
#SBATCH --account=edu                     # The account name for the job.
#SBATCH --job-name=tensor-parallel	  # The job name.
#SBATCH --gres=gpu:2             	  # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH -c 1                     	  # The number of cpu cores to use.
#SBATCH --time=0-01:00           	  # The time the job will take to run in D-HH:MM
#SBATCH --mem-per-cpu=5gb        	  # The memory the job will use per cpu core.

module load cuda

source ../../bin/activate

# For fast(er) debugging
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=30

torchrun --standalone --nproc_per_node=1 run_experiment.py tensor_parallel 10m
