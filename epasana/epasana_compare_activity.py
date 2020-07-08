import mne
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
images = utils.get_stimulus_images(stimuli, data_path='../data/epasana')
layer_outputs = model.get_layer_activations(images)
layer_activity = []
for output in layer_outputs:
    layer_activity = output.sum(dim=
