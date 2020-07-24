#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -t 10:00:00
#SBATCH --mem=32G
#SBATCH --output=vgg11_epasana-10kwords.out

module load anaconda
cp -r /m/nbe/scratch/reading_models/datasets/epasana-10kwords /tmp
cp -r /m/nbe/scratch/reading_models/datasets/epasana-nontext /tmp
#cp -r /m/nbe/scratch/reading_models/datasets/imagenet256 /tmp
#srun python ../train_net.py -a vgg11 --resume /m/nbe/scratch/reading_models/models/vgg11_imagenet.pth.tar --batch-size=128 --epochs=20 --lr=0.001 /tmp/epasana-10kwords /tmp/epasana-nontext /tmp/imagenet256 > vgg11_epasana-10kwords.log
srun python ../train_net.py -a vgg11 --resume /m/nbe/scratch/reading_models/models/vgg11_imagenet.pth.tar --batch-size=128 --epochs=20 --lr=0.001 /tmp/epasana-10kwords /tmp/epasana-nontext
#cp model_best.pth.tar ../data/models/vgg11_first_imagenet_then_epasana-10kwords_epasana-nontext_imagenet256.pth.tar
cp model_best.pth.tar ../data/models/vgg11_first_imagenet_then_epasana-10kwords_epasana-nontext.pth.tar
