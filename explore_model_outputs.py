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
model_name = 'vgg_first_imagenet64_then_tiny-words_tiny-consonants_tiny-symbols_tiny-imagenet_w2v'

# Get the images that were presented during the MEG experiment
stimuli = utils.get_stimulus_info(subject=2, data_path='M:/scratch/reading_models/')
images = utils.get_stimulus_images(subject=2, stimuli=stimuli, data_path='M:/scratch/reading_models/')

# Annotate the stimuli with the class labels in the tiny-words dataset. This
# dataset was used to train the model, so the model outputs correspond to these
# class labels.
with open('M:/scratch/reading_models/datasets/tiny-words/meta', 'rb') as f:
    meta = pickle.load(f)
labels = pd.DataFrame(meta['label_names']).reset_index()
labels.columns = ['class_index', 'text']
labels = labels.set_index('text')
stimuli = stimuli.join(labels)
order = np.argsort(stimuli[:180]['class_index'])
order = np.hstack([order, np.arange(180, 360)])

# Load the model and feed through the images
checkpoint = torch.load('M:/scratch/reading_models/models/%s.pth.tar' % model_name, map_location='cpu')
#model = networks.vgg.from_checkpoint(checkpoint)
#feature_outputs, classifier_outputs = model.get_layer_activations(images)
model = networks.vgg_sem.from_checkpoint(checkpoint, freeze=True)
feature_outputs, classifier_outputs, semantic_outputs = model.get_layer_activations(images)

# Plot all the images being fed into the model
plt.figure(figsize=(10, 10))
plt.imshow(make_grid(images/5 + 0.5, nrow=20).numpy().transpose(1, 2, 0))
plt.axis('off')
plt.tight_layout()

# Plot the model outputs
plt.figure()
plt.imshow(classifier_outputs[-1][order])
plt.axhline(180, color='black')
plt.axhline(180 + 90, color='black')
plt.colorbar()
plt.axvline(202, color='black')
plt.savefig('fig1.png', dpi=200)

plt.figure()
sem = semantic_outputs[-1][order]
val_range = abs(sem).max()
plt.imshow(sem, vmin=-val_range, vmax=val_range, cmap='RdBu_r')
plt.axhline(180, color='black')
plt.axhline(180 + 90, color='black')
plt.colorbar()
plt.savefig('fig2.png', dpi=200)
