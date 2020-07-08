import mne
import mne_rsa
import numpy as np
import networks
import torch
import pandas as pd

import utils

epochs = mne.read_epochs('../data/epasana/items-epo.fif')

model_name = 'vgg11_first_imagenet_then_epasana-words_epasana-nontext_imagenet256_w2v'
checkpoint = torch.load(f'../data/models/{model_name}.pth.tar', map_location='cpu')
model = networks.vgg_sem.from_checkpoint(checkpoint, freeze=True)

stimuli = pd.read_csv('stimulus_selection.csv')
images = utils.get_stimulus_images(stimuli, data_path='/m/nbe/scratch/epasana')
layer_outputs = model.get_layer_activations(images)
layer_activity = []

for output in layer_outputs:
    if output.ndim == 4:
        layer_activity.append(np.square(output).sum(axis=(1, 2, 3)))
    elif output.ndim == 2:
        layer_activity.append(np.square(output).sum(axis=1))


dsms = [mne_rsa.compute_dsm(a, metric='euclidean') for a in layer_activity]
mne_rsa.plot_dsms(dsms, names=['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'word', 'semantic'], n_rows=2)
