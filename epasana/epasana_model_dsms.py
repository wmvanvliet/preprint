import torch
import mne_rsa
import numpy as np
import networks
import pickle
from matplotlib import pyplot as plt
import pandas as pd

import utils

stimuli = pd.read_csv('stimulus_selection.csv')
images = utils.get_stimulus_images(stimuli, data_path='/m/nbe/scratch/epasana/')

model_name = 'vgg11_first_imagenet_then_epasana-words_epasana-nontext_imagenet256_w2v'
#model_name = 'vgg_first_imagenet64_then_tiny-words_tiny-consonants_tiny-symbols_tiny-imagenet'

checkpoint = torch.load('../data/models/%s.pth.tar' % model_name, map_location='cpu')
#model = networks.vgg.from_checkpoint(checkpoint, freeze=True)
#feature_outputs, classifier_outputs = model.get_layer_activations(images)
model = networks.vgg_sem.from_checkpoint(checkpoint, freeze=True)

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
    return x + (np.std(x) / 7) * np.random.randn(*x.shape)

print('Computing model DSMs...', end='', flush=True)
layer_outputs = model.get_layer_activations(images)
layer_activity = []
dsm_models = []
for output in layer_outputs:
    if output.ndim == 4:
        layer_activity.append(np.square(output).sum(axis=(1, 2, 3)))
    elif output.ndim == 2:
        layer_activity.append(np.square(output).sum(axis=1))
    dsm_models.append(mne_rsa.compute_dsm(n(output), metric='correlation'))
dsm_models += [
    mne_rsa.compute_dsm(stimuli[['type']], metric=words_only),
    mne_rsa.compute_dsm(stimuli[['type']], metric=letters_only),
    mne_rsa.compute_dsm(stimuli[['type']], metric=noise_level),
    mne_rsa.compute_dsm(images.numpy().reshape(len(images), -1), metric='euclidean'),
]
#dsm_names = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2', 'word', 
dsm_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'word', 'semantics',
             'Words only', 'Letters only', 'Noise level', 'Pixel distance']

with open(f'../data/dsms/epasana_{model_name}_dsms.pkl', 'wb') as f:
    pickle.dump(dict(dsms=dsm_models, dsm_names=dsm_names, layer_activity=layer_activity), f)

mne_rsa.plot_dsms(dsm_models, dsm_names, n_rows=3, cmap='magma')
plt.savefig(f'../figures/epasana_dsms_{model_name}.pdf')
