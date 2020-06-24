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
from pilot import utils

# The model to perform the analysis on. I keep changing this around as I train new models.
#model_name = 'vgg_first_imagenet64_then_tiny-words_tiny-consonants_tiny-imagenet'
model_name = 'vgg_first_imagenet_then_pilot-words_pilot-nontext_imagenet256'

# Get the images that were presented during the MEG experiment
stimuli = utils.get_stimulus_info(subject=2, data_path='data')
images = utils.get_stimulus_images(subject=2, stimuli=stimuli, data_path='data')

# Annotate the stimuli with the class labels in the tiny-words dataset. This
# dataset was used to train the model, so the model outputs correspond to these
# class labels.
meta = pd.read_csv('M:/scratch/reading_models/datasets/tiny-words/train.csv', index_col=0)

labels = meta.groupby('word').agg('first')['label']
stimuli = stimuli.join(labels)
order = np.argsort(stimuli[:180]['label'])
order = np.hstack([order, np.arange(180, 360)])

# Load the model and feed through the images
checkpoint = torch.load('data/models/%s.pth.tar' % model_name, map_location='cpu')
model = networks.vgg.from_checkpoint(checkpoint, freeze=True)
outputs = next(model.get_layer_activations(images, feature_layers=[], classifier_layers=[8]))
#model = networks.vgg_sem.from_checkpoint(checkpoint, freeze=True)
#feature_outputs, classifier_outputs, semantic_outputs = model.get_layer_activations(images)

# Plot all the images being fed into the model
plt.figure(figsize=(10, 10))
plt.imshow(make_grid(images/5 + 0.5, nrow=20).numpy().transpose(1, 2, 0))
plt.axis('off')
plt.tight_layout()

# Plot the model outputs
plt.figure()
plt.imshow(outputs[order])
plt.axhline(180, color='black')
plt.axhline(180 + 90, color='black')
plt.colorbar()
plt.axvline(202, color='black')
plt.savefig('fig1.png', dpi=200)

# plt.figure()
# sem = semantic_outputs[-1][order]
# val_range = abs(sem).max()
# plt.imshow(sem, vmin=-val_range, vmax=val_range, cmap='RdBu_r')
# plt.axhline(180, color='black')
# plt.axhline(180 + 90, color='black')
# plt.colorbar()
# plt.savefig('fig2.png', dpi=200)
