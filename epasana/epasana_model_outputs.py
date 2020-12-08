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

import sys
sys.path.append('../')

import networks
import utils
import dataloaders

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
#model = networks.vgg_sem.from_checkpoint(checkpoint, freeze=True)
output = next(model.get_layer_activations(images, feature_layers=[], classifier_layers=[6]))

# Translate output of the model to text predictions
predictions = stimuli.copy()
predictions['predicted_text'] = classes[output.argmax(axis=1)].values
predictions['predicted_class'] = output.argmax(axis=1)

predictions.to_csv('predictions.csv')

# How many of the word stimuli did we classify correctly?
word_predictions = predictions.query('type=="word"')
n_correct = (word_predictions['text'] == word_predictions['predicted_text']).sum()
accuracy = n_correct / len(word_predictions)
print(f'Word prediction accuracy: {n_correct}/{len(word_predictions)} = {accuracy * 100:.1f}%')  # 113/118, not bad at all!

# Write all predictions to large latex tables
for stimulus_type in ['word', 'pseudoword', 'consonants', 'symbols', 'noisy word']:
    with open(f'predictions_{stimulus_type.replace(" ", "_")}.tex', 'w') as f:
        f.write(predictions.query(f'type=="{stimulus_type}"').to_latex(index=False, columns=['type', 'text', 'predicted_text']))

