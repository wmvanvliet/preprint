#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH --mem=32G
#SBATCH --output=vgg11_epasana-1kwords.out

module load anaconda
cp -r /m/nbe/scratch/reading_models/datasets/epasana-1kwords /tmp
cp -r /m/nbe/scratch/reading_models/datasets/epasana-consonants /tmp
srun python ../train_net.py -a vgg11 --resume /m/nbe/scratch/reading_models/models/vgg11_imagenet256.pth.tar --batch-size=128 --epochs=20 --start-epoch=0 --lr=0.01 /tmp/epasana-1kwords /tmp/epasana-consonants > vgg11_epasana-1kwords.log && \
	cp model_best.pth.tar ../data/models/vgg11_first_imagenet_then_epasana-1kwords_epasana-consonants.pth.tar
