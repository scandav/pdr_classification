#!/bin/bash

#SBATCH --mail-user=davide.scandella@unibe.ch
#SBATCH --mail-type=FAIL,END
#SBATCH --account=ws_00000

# Job name
#SBATCH --job-name="the_lirio"
# Partition
#SBATCH --partition=gpu-invest # all, gpu, phi, long, gpu-invest

# Runtime and memory
#SBATCH --time=15:00:00    # days-HH:MM:SS
#SBATCH --mem-per-cpu=4G # it's memory PER CPU, NOT TOTAL RAM! maximum RAM is 246G in total
# total RAM is mem-per-cpu * cpus-per-task

# maximum cores is 20 on all, 10 on long, 24 on gpu, 64 on phi!
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1

#SBATCH --gres=gpu:rtx3090:1

#SBATCH --output=logs/slurm-%A_%a.out

# Main Python code below this line
module load Workspace
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate pdr_env
srun python pdr_classifier.py
