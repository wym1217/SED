#!/bin/bash

#SBATCH -p dgx2
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err


python run.py train_evaluate configs/baseline.yaml data/eval/wav.csv data/eval/label.csv 

