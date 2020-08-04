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
noise_path='noise.pdf'

def create_image_noise(data_path=data_path, index=300, 
                       n_noise_levels=100, max_noise_std=1.5):
    """Get the stimulus images presented during the MEG experiment with noises
    Parameters
    ----------
    data_path : str
        Path to the epasana dataset on scratch.
    index: int
        index of an image to generate noisy images based on
    n_noise_levels: int
        # of noise levels
    max_noise_std: float
        maximum level of noise specified by standard deviation of Gaussian

    Returns
    -------
    images : tensor, shape (n_noise_levels, n_colors, width, height)
        A tensor containing the bitmap data of an image presented to the
        subject during the MEG experiment with different noise levels.
        Ready to feed into a model.
    """
    stimuli = pd.read_csv(f'{data_path}/bids/stimuli.csv').iloc[index]


    class AddGaussianNoise(object):
        """ Function to add noises.
        """
        def __init__(self, mean=0., std=1.):
            self.std = std
            self.mean = mean

        def __call__(self, tensor):
            return (tensor + torch.randn(tensor.size()[1:]).expand(3, 224, 224)
                    * self.std + self.mean)

        def __repr__(self):
            return (self.__class__.__name__ + '(mean={0}, std={1})'
                    .format(self.mean, self.std))

    def make_preproc(noise_std):
        preproc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            AddGaussianNoise(0.5, noise_std)
        ])
        return preproc

    images = []
    fname = stimuli['fname']
    for noise_std in np.linspace(0, max_noise_std, n_noise_levels):
        with Image.open(f'{data_path}/bids/stimulus_images/{fname}') as orig:
            image = Image.new('RGB', (224, 224), '#696969')
            image.paste(orig, (12, 62))
            preproc = make_preproc(noise_std)
            image = preproc(image).unsqueeze(0)
            images.append(image)

    images = torch.cat(images, 0)
    return images

# create noisy images
img_noise = create_image_noise()

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
    outputs = model.get_layer_activations(img_noise,
                                          feature_layers=feature_layers,
                                          classifier_layers=classifier_layers)
    
# noisy image DSM
DSM_pixelwise_noise = mne_rsa.compute_dsm(img_noise, metric='euclidean')

# calculate RSA between model DSMs and image DSM
n_layers_to_plot = len(feature_layers) + len(classifier_layers) + (1 if arch=='vgg_sem' else 0)
dsms_noise = []
rsa_noise = []
for i in range(n_layers_to_plot):
    dsm = mne_rsa.compute_dsm(next(outputs), metric='correlation')
    rsa_noise.append(mne_rsa.rsa(dsm, DSM_pixelwise_noise))
    dsms_noise.append(dsm)
    del dsm

# plot model DSMs
h = int(np.ceil(np.sqrt(n_layers_to_plot)))
w = int(np.ceil(n_layers_to_plot/h))
figsize = (int(w*5), int(h*5))
fig, axes = plt.subplots(h, w, figsize=figsize)
for i in range(n_layers_to_plot-1, -1, -1): # loop through from the end to the first to reduce run time
    ax = axes.flat[i]
    sns.heatmap(distance.squareform(dsms_noise[i]), ax=ax, square=True)
    del dsms_noise[-1] # free memory

rsa_noise_round = np.round(rsa_noise, 2)    

feature = ("Feature Layer " + pd.Series(feature_layers, dtype=str)
           + " (RSA=" + pd.Series(rsa_noise_round[:len(feature_layers)], dtype=str) + ")")
classifier = ("Classifier Layer " + pd.Series(classifier_layers, dtype=str)
             + " (RSA=" + pd.Series(rsa_noise_round[len(feature_layers):len(feature_layers) + len(classifier_layers)], dtype=str) + ")")
if arch == 'vgg_sem':
    sem = "Semantics Layer (RSA=" + str(rsa_noise_round[-1])  + ")"

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

# save model DSMs for noisy imput
fig.savefig(noise_path)