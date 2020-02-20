#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH --mem=10G
#SBATCH --output=train_tiny-words.out

ml nvidia-pytorch
srun python ../train_net.py -a vgg --num-epochs=3 ../data/datasets/tiny-words > train_tiny-words.log
