import sys
sys.path.append('..')

import networks
import dataloaders
import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

# Load model
model_name = 'vgg11_first_imagenet_then_epasana-10kwords_noise'
checkpoint = torch.load(f'/m/nbe/scratch/reading_models/models/{model_name}.pth.tar', map_location='cuda')
model = networks.vgg11.from_checkpoint(checkpoint, freeze=True).cuda()

# Load data
transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

words = torch.utils.data.DataLoader(
    dataloaders.TFRecord('/m/nbe/scratch/reading_models/datasets/epasana-1kwords', train=False, transform=transform),
    batch_size=12,
    shuffle=False,
    pin_memory=True,
)

symbols = torch.utils.data.DataLoader(
    dataloaders.TFRecord('/m/nbe/scratch/reading_models/datasets/epasana-symbols', train=False, transform=transform),
    batch_size=12,
    shuffle=False,
    pin_memory=True,
)

# Run data through the model
feature_layers = [3]

word_activations = []
for batch, _ in tqdm(words, unit='batch'):
    batch = batch.cuda()
    for act in model.get_layer_activations(batch, feature_layers=feature_layers, classifier_layers=[], verbose=False):
        word_activations.append(act.mean(axis=(2, 3)))
word_activations = np.vstack(word_activations)

symbol_activations = []
for batch, _ in tqdm(symbols, unit='batch'):
    batch = batch.cuda()
    for act in model.get_layer_activations(batch, feature_layers=feature_layers, classifier_layers=[], verbose=False):
        symbol_activations.append(act.mean(axis=(2, 3)))
symbol_activations = np.vstack(symbol_activations)

##
# Plot one of the channels
channel = 0
fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
axes[0].hist(symbol_activations[:, channel], alpha=0.7, bins=100, color='C0')
axes[0].axvline(np.median(symbol_activations[:, channel]), color='C0')
axes[0].set_title('symbols')
axes[1].hist(word_activations[:, channel], alpha=0.7, bins=100, color='C1')
axes[1].axvline(np.median(word_activations[:, channel]), color='C1')
axes[1].set_title('words')
plt.tight_layout()

##
# Plot overall bias of each channel
bias = ttest_ind(np.log(word_activations), np.log(symbol_activations)).statistic
plt.figure()
plt.scatter(np.arange(word_activations.shape[1]), bias)
plt.axhline(0, color='black')
