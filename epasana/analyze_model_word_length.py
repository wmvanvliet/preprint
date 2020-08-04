#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:00:04 2020

@author: shimizt1
"""

import inspect
import torch
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import mne_rsa
from scipy.spatial import distance
import seaborn as sns
import pandas as pd
from PIL import Image
from reading_models import networks
import os
import contextlib

# parameters
model_path = ("/m/nbe/scratch/reading_models/models/"
              "vgg11_first_imagenet_then_epasana-10kwords_epasana-nontext.pth.tar") # need to select a model
data_path='/m/nbe/scratch/epasana'
meta_path = '/m/nbe/scratch/epasana/bids/stimuli.csv'
feature_layers=None # set to None to use default setting of "get_layer_activation" function or specify as a list
classifier_layers=None # set to None to use default setting of "get_layer_activation" function or specify as a list
len_path = "word_length.pdf"

def create_image_len(data_path=data_path, index=[300]):
    """Get the stimulus images presented during the MEG experiment
    Parameters
    ----------
    data_path : str
        Path to the epasana dataset on scratch.
    Index: list of int
        Indices of words with different lengths

    Returns
    -------
    images : tensor, shape (len(index), n_colors, width, height)
        A tensor containing the bitmap data of the images presented to the
        subject during the MEG experiment. Ready to feed into a model.
    """
    stimuli = pd.read_csv(f'{data_path}/bids/stimuli.csv').iloc[index]

    def make_preproc():
        preproc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return preproc

    images = []
    fnames = stimuli['fname']
    for fname in fnames:
        with Image.open(f'{data_path}/bids/stimulus_images/{fname}') as orig:
            image = Image.new('RGB', (224, 224), '#696969')
            image.paste(orig, (12, 62))
            preproc = make_preproc()
            image = preproc(image).unsqueeze(0)
            images.append(image)

    images = torch.cat(images, 0)
    return images

meta = pd.read_csv(meta_path)
lengths = np.array([len(word) for word in meta[meta.text.notna()].text])
unique_len = np.unique(lengths)

# pseudoword is chosen since other types do not have a word with 9 letters
# 'words' contain indices of 3 pseudowords with different lengths (7, 8, 9)
words = []
for l in unique_len:
    words_l = meta[meta.text.notna()][lengths == l]
    words.append(words_l[words_l.type=='pseudoword'].iloc[0].name)

# create images with different word lengths
img_len = create_image_len(index=words)

# Load the model and feed through the images
checkpoint = torch.load(model_path, map_location='cpu')

arch = checkpoint['arch']
if arch=='vgg11':
    model = networks.vgg11.from_checkpoint(checkpoint)
elif arch=='vgg_sem':
    model = networks.vgg_sem.from_checkpoint(checkpoint)
else:
    raise NotImplementedError("Model architecture is not expected; "
                              "currently, this function works for vgg_sem and vgg11.")
    
def get_default(func, param):
    signature = inspect.signature(func)
    return signature.parameters[param].default

if feature_layers is None:
    feature_layers = get_default(model.get_layer_activations, 'feature_layers')
if classifier_layers is None:
    classifier_layers = get_default(model.get_layer_activations, 'classifier_layers')
    
with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
    outputs = model.get_layer_activations(img_len,
                                          feature_layers=feature_layers,
                                          classifier_layers=classifier_layers)
    
# get image DSM
DSM_pixelwise_len = mne_rsa.compute_dsm(img_len, metric='euclidean')

# calculate RSA between model DSMs and image DSM
n_layers_to_plot = len(feature_layers) + len(classifier_layers) + (1 if arch=='vgg_sem' else 0)
dsms_len = []
rsa_len = []
for i in range(n_layers_to_plot):
    dsm = mne_rsa.compute_dsm(next(outputs), metric='correlation')
    rsa_len.append(mne_rsa.rsa(dsm, DSM_pixelwise_len))
    dsms_len.append(dsm)
    del dsm

# plot model DSMs
h = int(np.ceil(np.sqrt(n_layers_to_plot)))
w = int(np.ceil(n_layers_to_plot/h))
figsize = (int(w*5), int(h*5))
fig, axes = plt.subplots(h, w, figsize=figsize)
for i in range(n_layers_to_plot-1, -1, -1): # loop through from the end to the first to reduce run time
    ax = axes.flat[i]
    sns.heatmap(distance.squareform(dsms_len[i]), ax=ax, square=True)
    del dsms_len[-1] # free memory

rsa_len_round = np.round(rsa_len, 2)    

feature = ("Feature Layer " + pd.Series(feature_layers, dtype=str)
           + " (RSA=" + pd.Series(rsa_len_round[:len(feature_layers)], dtype=str) + ")")
classifier = ("Classifier Layer " + pd.Series(classifier_layers, dtype=str)
             + " (RSA=" + pd.Series(rsa_len_round[len(feature_layers):len(feature_layers) + len(classifier_layers)], dtype=str) + ")")
if arch == 'vgg_sem':
    sem = "Semantics Layer (RSA=" + str(rsa_len_round[-1])  + ")"

for i, f in enumerate(feature):
    axes.flat[i].title.set_text(f)

for i, c in enumerate(classifier):
    j = i + len(feature)
    axes.flat[j].title.set_text(c)

if arch == 'vgg_sem':
    axes.flat[-1].title.set_text(sem)

n_empty = (w * h) - n_layers_to_plot
for i in range(1, n_empty + 1):
    fig.delaxes(axes[-1, -i])

# save model DSMs for imput with different word lengths
fig.savefig(len_path)