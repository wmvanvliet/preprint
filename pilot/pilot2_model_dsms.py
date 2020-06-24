import torch
import mne_rsa
import numpy as np
import networks
import editdistance
import pickle
from matplotlib import pyplot as plt
from scipy.spatial import distance

import utils

stimuli = utils.get_stimulus_info(subject=2, data_path='../data')
images = utils.get_stimulus_images(subject=2, stimuli=stimuli, data_path='../data')

model_name = 'vgg_first_imagenet_then_pilot-words_pilot-nontext_imagenet256_w2v'
#model_name = 'vgg_first_imagenet64_then_tiny-words_tiny-consonants_tiny-symbols_tiny-imagenet'

checkpoint = torch.load('../data/models/%s.pth.tar' % model_name, map_location='cpu')
#model = networks.vgg.from_checkpoint(checkpoint, freeze=True)
#feature_outputs, classifier_outputs = model.get_layer_activations(images)
model = networks.vgg_sem.from_checkpoint(checkpoint, freeze=True)

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

def n(x):
    return x + (np.std(x) / 7) * np.random.randn(*x.shape)

print('Computing model DSMs...', end='', flush=True)
layer_outputs = model.get_layer_activations(images)
dsm_models = [mne_rsa.compute_dsm(n(output), metric='correlation') for output in layer_outputs]
dsm_models += [
    mne_rsa.compute_dsm(stimuli[['type']], metric=words_only),
    mne_rsa.compute_dsm(stimuli[['type']], metric=letters_only),
    mne_rsa.compute_dsm(stimuli[['noise_level']], metric='euclidean'),
    mne_rsa.compute_dsm(stimuli[['font']], metric=str_not_equal),
    mne_rsa.compute_dsm(stimuli[['rotation']], metric='sqeuclidean'),
    mne_rsa.compute_dsm(stimuli[['fontsize']], metric='sqeuclidean'),
    mne_rsa.compute_dsm(stimuli.index.tolist(), metric=str_dist),
    mne_rsa.compute_dsm(images.numpy().reshape(len(images), -1), metric='euclidean'),
]
#dsm_names = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2', 'word', 
dsm_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'word', 'semantics',
             'Words only', 'Letters only', 'Noise level', 'Font', 'Rotation', 'Fontsize', 'Edit distance', 'Pixel distance']

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
