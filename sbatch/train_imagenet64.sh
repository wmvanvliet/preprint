#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --mem=32G
#SBATCH --output=train_imagenet64.out

module load anaconda3
module load nvidia-pytorch
srun python ../train_net.py -a vgg --epochs=20 ../data/datasets/imagenet64 > train_imagenet64.log
