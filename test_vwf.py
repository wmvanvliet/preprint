import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import pandas as pd
from scipy.spatial import distance
import rsa
import mne
import numpy as np
import sys
from matplotlib import pyplot as plt

import networks

model_name = 'vgg_first_images_then_words3'

data_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder('/l/vanvlm1/tiny_word_image/val',
        transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(60),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    ),
    batch_size=256, shuffle=False, pin_memory=True)

checkpoint = torch.load('models/%s.pth.tar' % model_name)
model = networks.vgg(num_classes=400)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
