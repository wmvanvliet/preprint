"""
Visualize the output of the model. Main questions here are:

1) Can the model correctly classify the stimuli presented in the MEG experiment?
2) How does the model behave regarding consonant strings and symbol strings?
"""
import torch
import numpy as np
import pickle
import pandas as pd
import mkl
mkl.set_num_threads(4)

import sys
sys.path.append('../')

import networks
import utils
import dataloaders
from config import fname

# The model to perform the analysis on. I keep changing this around as I train new models.
model_name = 'vgg11_first_imagenet_then_epasana-10kwords_noise'
classes = dataloaders.TFRecord('/m/nbe/scratch/reading_models/datasets/epasana-10kwords').classes
classes.append(pd.Series(['noise'], index=[10000]))

# Get the images that were presented during the MEG experiment
stimuli = pd.read_csv('stimulus_selection.csv')
images = utils.get_stimulus_images(stimuli, data_path='/m/nbe/scratch/epasana')

# Load the model and feed through the images
checkpoint = torch.load('../data/models/%s.pth.tar' % model_name, map_location='cpu')
model = networks.vgg11.from_checkpoint(checkpoint, freeze=True)

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

layer_names = [
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
]
with open(fname.layer_activity(model=model_name), 'wb') as f:
    pickle.dump(dict(layer_names=layer_names, layer_activity=layer_activity), f)

# Translate output of the model to text predictions
predictions = stimuli.copy()
predictions['predicted_text'] = classes[layer_activity[-1].argmax(axis=1)].values
predictions['predicted_class'] = layer_activity[-1].argmax(axis=1)
predictions.to_csv('predictions.csv')

# How many of the word stimuli did we classify correctly?
word_predictions = predictions.query('type=="word"')
n_correct = (word_predictions['text'] == word_predictions['predicted_text']).sum()
accuracy = n_correct / len(word_predictions)
print(f'Word prediction accuracy: {n_correct}/{len(word_predictions)} = {accuracy * 100:.1f}%')  # 113/118, not bad at all!
