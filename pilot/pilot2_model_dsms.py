import torch
import torchvision.transforms as transforms
import rsa
import mne
import pandas as pd
import numpy as np
from os.path import getsize
from PIL import Image
from tqdm import tqdm
import networks
import editdistance
import pickle
from matplotlib import pyplot as plt
from scipy.spatial import distance

import utils

stimuli = utils.get_stimulus_info(subject=2)
images = utils.get_stimulus_images(subject=2, stimuli=stimuli)

#model_name = 'vgg_first_imagenet64_then_tiny-words-noisy_tiny-imagenet'
model_name = 'vgg_first_imagenet64_then_tiny-words_tiny-imagenet'
#model_name = 'vgg_first_imagenet64_then_tiny-words_tiny-consonants_w2v'
#model_name = 'vgg_first_imagenet64_then_tiny-words_tiny-imagenet_then_w2v'

checkpoint = torch.load('../data/models/%s.pth.tar' % model_name, map_location='cpu')
#model = networks.vgg.from_checkpoint(checkpoint, freeze=True)
#feature_outputs, classifier_outputs = model.get_layer_activations(images)
model = networks.vgg_sem.from_checkpoint(checkpoint, freeze=True)
feature_outputs, classifier_outputs, semantic_outputs = model.get_layer_activations(images)

def words_only(x, y):
    if (x != 'word' or y != 'word'):
        return 1
    else:
        return 0

def letters_only(x, y):
    if (x == 'symbols' or y == 'symbols'):
        return 1
    else:
        return 0

def str_not_equal(x, y):
    if x == y:
        return 0
    else:
        return 1

def str_dist(x, y):
    return editdistance.eval(x[0], y[0])

print('Computing model DSMs...', end='', flush=True)
dsm_models = [
    rsa.compute_dsm(feature_outputs[0], metric='correlation'),
    rsa.compute_dsm(feature_outputs[1], metric='correlation'),
    rsa.compute_dsm(feature_outputs[2], metric='correlation'),
    rsa.compute_dsm(feature_outputs[3], metric='correlation'),
    rsa.compute_dsm(classifier_outputs[0], metric='correlation'),
    rsa.compute_dsm(classifier_outputs[1], metric='correlation'),
    rsa.compute_dsm(classifier_outputs[2], metric='correlation'),
    rsa.compute_dsm(semantic_outputs[0], metric='correlation'),
    rsa.compute_dsm(stimuli[['type']], metric=words_only),
    rsa.compute_dsm(stimuli[['type']], metric=letters_only),
    rsa.compute_dsm(stimuli[['noise_level']], metric='euclidean'),
    rsa.compute_dsm(stimuli[['font']], metric=str_not_equal),
    rsa.compute_dsm(stimuli[['rotation']], metric='sqeuclidean'),
    rsa.compute_dsm(stimuli[['fontsize']], metric='sqeuclidean'),
    rsa.compute_dsm(stimuli.index.tolist(), metric=str_dist),
    rsa.compute_dsm(images.numpy().reshape(len(images), -1), metric='euclidean'),
    #rsa.compute_dsm(abs(feature_outputs[0]).sum(axis=(1, 2, 3), keepdims=True), metric='euclidean'),
    #rsa.compute_dsm(abs(feature_outputs[1]).sum(axis=(1, 2, 3), keepdims=True), metric='euclidean'),
    #rsa.compute_dsm(abs(feature_outputs[2]).sum(axis=(1, 2, 3), keepdims=True), metric='euclidean'),
    #rsa.compute_dsm(abs(feature_outputs[3]).sum(axis=(1, 2, 3), keepdims=True), metric='euclidean'),
    #rsa.compute_dsm(abs(classifier_outputs[0]).sum(axis=1, keepdims=True), metric='euclidean'),
    #rsa.compute_dsm(abs(classifier_outputs[1]).sum(axis=1, keepdims=True), metric='euclidean'),
    #rsa.compute_dsm(abs(classifier_outputs[2]).sum(axis=1, keepdims=True), metric='euclidean'),
]
dsm_names = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2', 'word', 'semantics',
             'Words only', 'Letters only', 'Noise level', 'Font', 'Rotation', 'Fontsize', 'Edit distance', 'Pixel distance']
#             'conv1_act', 'conv2_act', 'conv3_act', 'conv4_act', 'fc1_act', 'fc2_act', 'output_act']

with open(f'../data/dsms/pilot2_{model_name}_dsms.pkl', 'wb') as f:
    pickle.dump(dict(dsms=dsm_models, dsm_names=dsm_names), f)

n_rows = 4
n_cols = 4
fig, ax = plt.subplots(n_rows, n_cols, sharex='col', sharey='row', figsize=(10, 10))
for row in range(n_rows):
    for col in range(n_cols):
        i = row * n_cols + col
        if i < len(dsm_models):
            ax[row, col].imshow(distance.squareform(dsm_models[i]), cmap='magma')
            ax[row, col].set_title(dsm_names[i])
plt.tight_layout()
plt.savefig(f'../figures/pilot2_dsms_{model_name}.pdf')
