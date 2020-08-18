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
import os
import contextlib

# We're in a subdirectory, so add the main directory to the path so we can
# import from it
import sys
sys.path.append('..')
import networks

# parameters
model_path = ("/m/nbe/scratch/reading_models/models/"
              "vgg11_first_imagenet_then_epasana-10kwords_epasana-nontext_imagenet256.pth.tar") # need to select a model
data_path='/m/nbe/scratch/epasana'
meta_path = '/m/nbe/scratch/epasana/bids/stimuli.csv'
feature_layers=None # set to None to use default setting of "get_layer_activation" function or specify as a list
classifier_layers=None # set to None to use default setting of "get_layer_activation" function or specify as a list
rot_path='rotation.pdf'

def create_image_rotation(data_path=data_path, index=300, 
                          angles=[0, 45, -45, 90, -90, 135, -135, 180]):
    """Get the stimulus images presented during the MEG experiment with rotation.
    Parameters
    ----------
    data_path : str
        Path to the epasana dataset on scratch.
    index: int
        index of an image to generate rotated images based on
    angles: list of int
        angles to rotate

    Returns
    -------
    images : tensor, shape (len(angles), n_colors, width, height)
        A tensor containing the bitmap data of an image presented to the
        subject during the MEG experiment with rotations.
        Ready to feed into a model.
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
    fname = stimuli['fname']
    for angle in angles:
        with Image.open(f'{data_path}/bids/stimulus_images/{fname}') as orig:
            image = Image.new('RGB', (224, 224), '#696969')
            image_w, image_h = image.size
            orig = orig.rotate(angle, expand=True, fillcolor='#696969')
            orig_w, orig_h = orig.size
            image.paste(orig, ((image_w - orig_w) // 2, (image_h - orig_h) // 2))
            preproc = make_preproc()
            image = preproc(image).unsqueeze(0)
            images.append(image)

    images = torch.cat(images, 0)
    return images

# create rotated images
img_rotation = create_image_rotation()

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
    outputs = model.get_layer_activations(img_rotation,
                                          feature_layers=feature_layers,
                                          classifier_layers=classifier_layers)
    
# get image DSM
DSM_pixelwise_rot = mne_rsa.compute_dsm(img_rotation, metric='euclidean')

# calculate RSA between model DSMs and image DSM
n_layers_to_plot = len(feature_layers) + len(classifier_layers) + (1 if arch=='vgg_sem' else 0)
dsms_rot = []
rsa_rot = []
for i in range(n_layers_to_plot):
    dsm = mne_rsa.compute_dsm(next(outputs), metric='correlation')
    rsa_rot.append(mne_rsa.rsa(dsm, DSM_pixelwise_rot))
    dsms_rot.append(dsm)
    del dsm

# plot model DSMs
h = int(np.ceil(np.sqrt(n_layers_to_plot)))
w = int(np.ceil(n_layers_to_plot/h))
figsize = (int(w*5), int(h*5))
fig, axes = plt.subplots(h, w, figsize=figsize)
for i in range(n_layers_to_plot-1, -1, -1): # loop through from the end to the first to reduce run time
    ax = axes.flat[i]
    sns.heatmap(distance.squareform(dsms_rot[i]), ax=ax, square=True)
    del dsms_rot[-1] # free memory

rsa_rot_round = np.round(rsa_rot, 2)    

feature = ("Feature Layer " + pd.Series(feature_layers, dtype=str)
           + " (RSA=" + pd.Series(rsa_rot_round[:len(feature_layers)], dtype=str) + ")")
classifier = ("Classifier Layer " + pd.Series(classifier_layers, dtype=str)
             + " (RSA=" + pd.Series(rsa_rot_round[len(feature_layers):len(feature_layers) + len(classifier_layers)], dtype=str) + ")")
if arch == 'vgg_sem':
    sem = "Semantics Layer (RSA=" + str(rsa_rot_round[-1])  + ")"

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

# save model DSMs for rotated imput
fig.savefig(rot_path)
