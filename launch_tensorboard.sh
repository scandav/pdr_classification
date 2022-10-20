#!/bin/bash
module load Workspace
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate pdr_env
# tensorboard --logdir .
tensorboard --logdir runs