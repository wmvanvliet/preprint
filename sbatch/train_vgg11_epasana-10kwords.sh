#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --mem=32G
#SBATCH --output=vgg11_epasana-10kwords.out

module load anaconda
cp -r /m/nbe/scratch/reading_models/datasets/epasana-10kwords /tmp
#cp -r /m/nbe/scratch/reading_models/datasets/epasana-nontext /tmp
#cp -r /m/nbe/scratch/reading_models/datasets/imagenet256 /tmp
cp -r /m/nbe/scratch/reading_models/datasets/noise /tmp
#srun python ../train_net.py -a vgg11 --resume /m/nbe/scratch/reading_models/models/vgg11_imagenet256.pth.tar --batch-size=128 --epochs=20 --start-epoch=0 --lr=0.001 /tmp/epasana-10kwords /tmp/epasana-consonants /tmp/epasana-symbols /tmp/imagenet256 > vgg11_epasana-10kwords.log
#srun python ../train_net.py -a vgg11 --resume /m/nbe/scratch/reading_models/models/vgg11_imagenet256.pth.tar --batch-size=128 --epochs=20 --start-epoch=0 --lr=0.001 /tmp/epasana-10kwords /tmp/epasana-nontext /tmp/imagenet256 > vgg11_epasana-10kwords.log
#srun python ../train_net.py -a vgg11 --resume /m/nbe/scratch/reading_models/models/vgg11_imagenet.pth.tar --batch-size=128 --epochs=20 --start-epoch=0 --lr=0.001 /tmp/epasana-10kwords /tmp/epasana-nontext > vgg11_epasana-10kwords.log
#srun python ../train_net.py -a vgg11 --resume /m/nbe/scratch/reading_models/models/vgg11_imagenet.pth.tar --batch-size=128 --epochs=20 --start-epoch=0 --lr=0.001 /tmp/epasana-10kwords /tmp/epasana-nontext /tmp/epasana-noise > vgg11_epasana-10kwords.log
srun python ../train_net.py -a vgg11 --resume /m/nbe/scratch/reading_models/models/vgg11_imagenet.pth.tar --batch-size=128 --epochs=20 --start-epoch=0 --lr=0.001 /tmp/epasana-10kwords /tmp/noise > vgg11_epasana-10kwords.log
#srun python ../train_net.py -a vgg11 --resume /m/nbe/scratch/reading_models/models/vgg11_imagenet.pth.tar --batch-size=128 --epochs=20 --start-epoch=0 --lr=0.01 /tmp/epasana-10kwords > vgg11_epasana-10kwords.log
#srun python ../train_net.py -a vgg11 --batch-size=128 --epochs=20 --start-epoch=0 --lr=0.01 /tmp/epasana-10kwords > vgg11_epasana-10kwords.log
#srun python ../train_net.py -a vgg11 --batch-size=128 --epochs=20 --start-epoch=0 --lr=0.01 /tmp/epasana-10kwords /tmp/noise > vgg11_epasana-10kwords.log

#cp model_best.pth.tar ../data/models/vgg11_first_imagenet_then_epasana-10kwords_epasana-consonants_epasana-symbols_imagenet256.pth.tar
#cp model_best.pth.tar ../data/models/vgg11_first_imagenet_then_epasana-10kwords_epasana-nontext_imagenet256.pth.tar
#cp model_best.pth.tar ../data/models/vgg11_first_imagenet_then_epasana-10kwords_epasana-nontext.pth.tar
#cp model_best.pth.tar ../data/models/vgg11_first_imagenet_then_epasana-10kwords_epasana-nontext_epasana-noise.pth.tar
#cp model_best.pth.tar ../data/models/vgg11_first_imagenet_then_epasana-10kwords.pth.tar
cp model_best.pth.tar ../data/models/vgg11_first_imagenet_then_epasana-10kwords_noise.pth.tar
#cp model_best.pth.tar ../data/models/vgg11_epasana-10kwords.pth.tar
#cp model_best.pth.tar ../data/models/vgg11_epasana-10kwords_noise.pth.tar
