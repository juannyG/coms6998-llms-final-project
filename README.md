# coms6998-llms-final-project

## Setting up the python environment
While HPC likes to use `conda`, creating a new environment seems to be broken. Try if you can and we can update this later.
So far, I was able to set up a virtualenv and install packages in it:

```sh
# Connect to insomnia
ssh <UNI>@insomnia.rcs.columbia.edu

# Move to a compute node (if you want a GPU, add --gres=gpu:1 - useful if you want to use things like nvidia-smi)
srun --pty -t 0-2:00 -A edu /bin/bash

# Move into your scratch space and set up the environment
cd /insomnia001/depts/edu/COMS-E6998-015/<UNI>

# Create the env dir; default name provided
/insomnia001/shared/apps/anaconda/2023.09/bin/python -m venv sllm-final-project-env
cd sllm-final-project-env
source bin/activate
pip install -r requirements.txt
```
