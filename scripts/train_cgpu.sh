#!/bin/bash
#SBATCH -C gpu
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH -J train-cgpu
#SBATCH -d singleton
#SBATCH -o logs/%x-%j.out

# Setup software
. scripts/setup_cgpu.sh

# Run training
srun --ntasks-per-node 8 -l -u \
    python train.py -d $@
