"""
Visualize the output of the model. Main questions here are:

1) Can the model correctly classify the stimuli presented in the MEG experiment?
2) How does the model behave regarding consonant strings and symbol strings?
"""
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
import mne
import numpy as np
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import mkl
mkl.set_num_threads(4)

import networks
import utils

# The model to perform the analysis on. I keep changing this around as I train new models.
model_name = 'vgg11_first_imagenet_then_epasana-10kwords_epasana-nontext'

# Get the images that were presented during the MEG experiment
stimuli = pd.read_csv('stimulus_selection.csv')
images = utils.get_stimulus_images(stimuli, data_path='/m/nbe/scratch/epasana')

# Load the model and feed through the images
checkpoint = torch.load('../data/models/%s.pth.tar' % model_name, map_location='cpu')
model = networks.vgg11.from_checkpoint(checkpoint, freeze=True)
#model = networks.vgg_sem.from_checkpoint(checkpoint, freeze=True)
outputs = model.get_layer_activations(images, feature_layers=[], classifier_layers=[4, 7])
fc2 = next(outputs)
outputs = next(outputs)

# Plot all the images being fed into the model
plt.figure(figsize=(10, 10))
plt.imshow(make_grid(images/5 + 0.5, nrow=20).numpy().transpose(1, 2, 0))
plt.axis('off')
plt.tight_layout()

# Plot the model outputs
plt.figure()
plt.imshow(outputs)
#plt.axhline(180, color='black')
#plt.axhline(180 + 90, color='black')
plt.colorbar()
#plt.axvline(202, color='black')
#plt.savefig('fig1.png', dpi=200)
plt.xlim(0, 500)
#plt.ylim(352, 470)

# plt.figure()
# sem = semantic_outputs[-1][order]
# val_range = abs(sem).max()
# plt.imshow(sem, vmin=-val_range, vmax=val_range, cmap='RdBu_r')
# plt.axhline(180, color='black')
# plt.axhline(180 + 90, color='black')
# plt.colorbar()
# plt.savefig('fig2.png', dpi=200)
