# coms6998-llms-final-project

Examples:
```sh
# Run a single GPU experiment with 10m paramemters
python run_experiment.py single_gpu 10m

# "Dry run" the experiment - only print out model param info
python run_experiment.py --dry-run single_gpu cpu
```

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

# Clone the repo and install reqs
git clone git@github.com:juannyG/coms6998-llms-final-project.git proj
cd proj
pip install -r requirements.txt
```

# Tips/Tricks
## .bashrc "setup" function
Add the following blurb to your `$HOME/.bashrc`:
```sh
setup_venv() {
    cd /insomnia001/depts/edu/COMS-E6998-015/<UNI>/sllm-final-project-env
    source bin/activate
    cd proj
    pip install -r requirements.txt
}
```

After you login and move to a compute node in HPC you can then run:
```sh
source .bashrc
setup_venv
```
This will put you in the repo directory with virtualenv already activated.

# TODO: How to run using Makefile
