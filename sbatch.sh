#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH --job-name=lp
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
bash train.sh