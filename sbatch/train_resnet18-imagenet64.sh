#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --mem=16G
#SBATCH --output=train_resnet18-imagenet64.out

module load anaconda3
srun python ../train_net.py -a resnet18 --epochs=20 ../data/datasets/imagenet64 > train_resnet18-imagenet64.log
