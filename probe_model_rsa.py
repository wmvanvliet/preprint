"""
Probe the computational model using representational similarity analysis (RSA).

Right now I'm using the images presented in the MEG experiment to conduct the
RSA analysis. This is not optimal. I should generate custom images to probe a
single aspect of the model.
"""
import torch
import numpy as np
from matplotlib import pyplot as plt
import rsa
import editdistance

import networks
from pilot import utils

# Get the images that were presented during the MEG experiment
stimuli = utils.get_stimulus_info(subject=2)
images = utils.get_stimulus_images(subject=2, stimuli=stimuli)

# Load the model and feed through the images
model_name = 'vgg_first_imagenet64_then_tiny-words-noisy_tiny-imagenet'
checkpoint = torch.load('/m/nbe/scratch/reading_models/models/%s.pth.tar' % model_name, map_location='cpu')
model = networks.vgg.from_checkpoint(checkpoint)
feature_outputs, classifier_outputs = model.get_layer_activations(images)

# Create different DSMs to probe the model using RSA
def words_only(x, y):
    """Distance function to generate a word-selective DSM"""
    if (x != 'word' or y != 'word') and (x != y):
        return 1
    else:
        return 0

def letters_only(x, y):
    """Distance function to generate a letter-selective DSM"""
    if (x == 'symbols' or y == 'symbols') and (x != y):
        return 1
    else:
        return 0

def not_equal(x, y):
    """Distance function based on equals"""
    if x == y:
        return 0
    else:
        return 1

def str_dist(x, y):
    """Edit distance metric"""
    return editdistance.eval(x[0], y[0])

print('Computing DSMs...', end='', flush=True)
dsms_network = [
    rsa.compute_dsm(feature_outputs[0], metric='correlation'),
    rsa.compute_dsm(feature_outputs[1], metric='correlation'),
    rsa.compute_dsm(feature_outputs[2], metric='correlation'),
    rsa.compute_dsm(feature_outputs[3], metric='correlation'),
    rsa.compute_dsm(classifier_outputs[0], metric='correlation'),
    rsa.compute_dsm(classifier_outputs[1], metric='correlation'),
    rsa.compute_dsm(classifier_outputs[2], metric='correlation'),
]
dsms_model = [
    rsa.compute_dsm(stimuli[['type']], metric=words_only),
    rsa.compute_dsm(stimuli[['type']], metric=letters_only),
    rsa.compute_dsm(stimuli[['noise_level']], metric='sqeuclidean'),
    rsa.compute_dsm(stimuli[['font']], metric=not_equal),
    rsa.compute_dsm(stimuli[['rotation']], metric='sqeuclidean'),
    rsa.compute_dsm(stimuli[['fontsize']], metric='sqeuclidean'),
    rsa.compute_dsm(stimuli.index.tolist(), metric=str_dist),
    rsa.compute_dsm(images.numpy().reshape(len(images), -1), metric='sqeuclidean'),
    rsa.compute_dsm(images.numpy().reshape(len(images), -1).sum(axis=1), metric='sqeuclidean'),
]
dsms_names = ['Words only', 'Letters only', 'Noise level', 'Font', 'Rotation',
              'Fontsize', 'Edit distance', 'Pixel distance', 'Pixel count']

# Perform the RSA analysis
rsa_results = rsa.rsa(dsms_model, dsms_network, metric='kendall-tau-a')
n_models, n_layers = rsa_results.shape

# Plot the results
f = plt.figure(figsize=(8, 5))
axs = f.subplots(int(np.ceil(n_models / 5)), 5, sharex=True, sharey=True)
for i, (name, result) in enumerate(zip(dsms_names, rsa_results)):
    ax = axs[i // 5, i % 5]
    ax.bar(np.arange(n_layers), result)
    ax.axhline(0, color='black')
    ax.set_title(name)
    if (i // 5) == len(axs) - 1:
        ax.set_xlabel('Network layer')
        ax.set_xticks(np.arange(n_layers))
        ax.set_xticklabels(['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2', 'output'], rotation=90)
plt.tight_layout()
