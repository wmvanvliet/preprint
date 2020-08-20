"""
Compute DSMs from the model activations for use in RSA with the epasana dataset.
"""
import torch
import mne_rsa
import numpy as np
import networks
import pickle
from matplotlib import pyplot as plt
import pandas as pd

import utils

torch.set_num_threads(1)

stimuli = pd.read_csv('stimulus_selection.csv')
images = utils.get_stimulus_images(stimuli)

model_name = 'vgg11_first_imagenet_then_epasana-10kwords_epasana-nontext'


checkpoint = torch.load('../data/models/%s.pth.tar' % model_name, map_location='cpu')
model = networks.vgg11.from_checkpoint(checkpoint, freeze=True)

def words_only(x, y):
    if (x not in ['word', 'pseudoword'] or y not in ['word', 'pseudoword']):
        return 1
    else:
        return 0

def letters_only(x, y):
    if (x in ['symbols', 'noisy word'] or y in ['symbols', 'noisy word']):
        return 1
    else:
        return 0

def noise_level(x, y):
    if (x != 'noisy word' or y != 'noisy word'):
        return 1
    else:
        return 0

def n(x):
    return x
    #return x + (np.std(x) / 7) * np.random.randn(*x.shape)

print('Computing model DSMs...', end='', flush=True)
layer_outputs = model.get_layer_activations(
    images,
    feature_layers=[0, 1, 4, 5, 11, 12, 18, 19, 25, 26],
    classifier_layers=[0, 1, 3, 4, 6, 7]
)
layer_activity = []
dsm_models = []
for output in layer_outputs:
    if output.ndim == 4:
        layer_activity.append(np.square(output).mean(axis=(1, 2, 3)))
    elif output.ndim == 2:
        layer_activity.append(np.square(output).mean(axis=1))
    if output.shape[-1] == 10001:
        print('Removing nontext class')
        output = np.hstack((output[:, :10000], output[:, 10001:]))
        print('New output shape:', output.shape)
    dsm_models.append(mne_rsa.compute_dsm(n(output), metric='correlation'))

dsm_models += [
    mne_rsa.compute_dsm(stimuli[['type']], metric=words_only),
    mne_rsa.compute_dsm(stimuli[['type']], metric=letters_only),
    mne_rsa.compute_dsm(stimuli[['type']], metric=noise_level),
    mne_rsa.compute_dsm(images.numpy().reshape(len(images), -1), metric='euclidean'),
]

dsm_names = [
    'conv1',
    'conv1_relu',
    'conv2',
    'conv2_relu',
    'conv3',
    'conv3_relu',
    'conv4',
    'conv4_relu',
    'conv5',
    'conv5_relu',
    'fc1',
    'fc1_relu',
    'fc2',
    'fc2_relu',
    'word',
    'word_relu',
    'Words only',
    'Letters only',
    'Noise level',
    'Pixel distance'
]

with open(f'../data/dsms/epasana_{model_name}_dsms.pkl', 'wb') as f:
    pickle.dump(dict(dsms=dsm_models, dsm_names=dsm_names, layer_activity=layer_activity), f)

mne_rsa.plot_dsms(dsm_models, dsm_names, n_rows=3, cmap='magma')
plt.savefig(f'../figures/epasana_dsms_{model_name}.pdf')
