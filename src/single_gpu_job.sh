#!/bin/sh
#
#SBATCH --account=edu                     # The account name for the job.
#SBATCH --job-name=single-gpu-baseline    # The job name.
#SBATCH --gres=gpu:1             	  # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH -c 1                     	  # The number of cpu cores to use.
#SBATCH --time=0-01:00           	  # The time the job will take to run in D-HH:MM
#SBATCH --mem-per-cpu=5gb        	  # The memory the job will use per cpu core.

module load cuda

source ../../bin/activate
python run_experiment.py single_gpu 10m
