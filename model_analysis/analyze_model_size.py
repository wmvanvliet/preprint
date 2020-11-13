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
import numpy.random as npr
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import mne_rsa
from scipy.spatial import distance
import seaborn as sns
import pandas as pd
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
feature_layers=None # set to None to use default setting of "get_layer_activation" function or specify as a list
classifier_layers=None # set to None to use default setting of "get_layer_activation" function or specify as a list
size_path='size.pdf'

def create_image(word='NAVETTA', rotation=0, fontsize=20, font='arial', noise_level=0,
                 data_path = '/m/nbe/scratch/reading_models'):
    """Get the stimulus images presented during the MEG experiment with different text sizes
    Parameters
    ----------
    data_path : str
        Path to the epasana dataset on scratch.

    Returns
    -------
    image : tensor
    """
    fonts = {
        'ubuntu mono': [None, f'{data_path}/fonts/UbuntuMono-R.ttf'],
        'courier': [None, f'{data_path}/fonts/courier.ttf'],
        'luxi mono regular': [None, f'{data_path}/fonts/luximr.ttf'],
        'lucida console': [None, f'{data_path}/fonts/LucidaConsole-R.ttf'],
        'lekton': [None, f'{data_path}/fonts/Lekton-Regular.ttf'],
        'dejavu sans mono': [None, f'{data_path}/fonts/DejaVuSansMono.ttf'],
        'times new roman': [None, f'{data_path}/fonts/times.ttf'],
        'arial': [None, f'{data_path}/fonts/arial.ttf'],
        'arial black': [None, f'{data_path}/fonts/arialbd.ttf'],
        'verdana': [None, f'{data_path}/fonts/verdana.ttf'],
        'comic sans ms': [None, f'{data_path}/fonts/comic.ttf'],
        'georgia': [None, f'{data_path}/fonts/georgia.ttf'],
        'liberation serif': [None, f'{data_path}/fonts/LiberationSerif-Regular.ttf'],
        'impact': [None, f'{data_path}/fonts/impact.ttf'],
        'roboto condensed': [None, f'{data_path}/fonts/Roboto-Light.ttf'],
    }
    
    dpi = 96.
    
    f = Figure(figsize=(224 / dpi, 224 / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(f)
    
    f.clf()
    ax = f.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.fill_between([0, 1], [1, 1], color='#696969')
    
    ax = f.add_axes([0, 0, 1, 1], label='text')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fontfamily, fontfile = fonts[font]
    fontprop = fm.FontProperties(family=fontfamily, fname=fontfile, size=fontsize)
    ax.text(0.5, 0.5, word, ha='center', va='center', rotation=rotation,
            fontproperties=fontprop, alpha=1 - noise_level)
    
    ax = f.add_axes([0, 0, 1, 1], label='noise')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    noise = npr.rand(224, 224)
    ax.imshow(noise, extent=[0, 1, 0, 1], cmap='gray', alpha=noise_level)
    
    canvas.draw()
    buffer, (width, height) = canvas.print_to_buffer()
    image = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))[:, :, :3]
    
    return image

def create_image_size(word='NAVETTA', size_lim = [10, 30], n_size_levels=100,
                      rotation=0, font='arial', noise_level=0,
                      data_path='/m/nbe/scratch/reading_models'):
    """Get the stimulus images presented during the MEG experiment with different text sizes
    Parameters
    ----------
    data_path : str
        Path to the epasana dataset on scratch.
    size_lim: list
        List of max and min text size
    n_size_levels: int
        # of size levels

    Returns
    -------
    images : tensor, shape (n_size_levels, n_colors, width, height)
        A tensor containing the bitmap data of an image presented to the
        subject during the MEG experiment with different text sizes.
        Ready to feed into a model.
    """
    def make_preproc():
        preproc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return preproc
    
    preproc = make_preproc()
    
    sizes = np.linspace(size_lim[0], size_lim[1], n_size_levels)
    
    images = []
    for size in sizes:
        img_np = create_image(word=word, rotation=rotation, fontsize=size, font=font,
                              noise_level=noise_level, data_path=data_path)
        
        image = preproc(img_np)
        image = image.unsqueeze(0)
        images.append(image)
        
    images = torch.cat(images, 0)
    
    return images

# create images with different text size
img_size = create_image_size()

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
    outputs = model.get_layer_activations(img_size,
                                          feature_layers=feature_layers,
                                          classifier_layers=classifier_layers)
    
# DSM of images with different text size
DSM_pixelwise_size = mne_rsa.compute_dsm(img_size, metric='euclidean')

# calculate RSA between model DSMs and image DSM
n_layers_to_plot = len(feature_layers) + len(classifier_layers) + (1 if arch=='vgg_sem' else 0)
dsms_size = []
rsa_size = []
for i in range(n_layers_to_plot):
    dsm = mne_rsa.compute_dsm(next(outputs), metric='correlation')
    rsa_size.append(mne_rsa.rsa(dsm, DSM_pixelwise_size))
    dsms_size.append(dsm)
    del dsm

# plot model DSMs
h = int(np.ceil(np.sqrt(n_layers_to_plot)))
w = int(np.ceil(n_layers_to_plot/h))
figsize = (int(w*5), int(h*5))
fig, axes = plt.subplots(h, w, figsize=figsize)
for i in range(n_layers_to_plot-1, -1, -1): # loop through from the end to the first to reduce run time
    ax = axes.flat[i]
    sns.heatmap(distance.squareform(dsms_size[i]), ax=ax, square=True)
    del dsms_size[-1] # free memory

rsa_size_round = np.round(rsa_size, 2)    

feature = ("Feature Layer " + pd.Series(feature_layers, dtype=str)
           + " (RSA=" + pd.Series(rsa_size_round[:len(feature_layers)], dtype=str) + ")")
classifier = ("Classifier Layer " + pd.Series(classifier_layers, dtype=str)
              + " (RSA=" + pd.Series(rsa_size_round[len(feature_layers):len(feature_layers) + len(classifier_layers)], dtype=str) + ")")
if arch == 'vgg_sem':
    sem = "Semantics Layer (RSA=" + str(rsa_size_round[-1])  + ")"

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

# save model DSMs for imput with different text size
fig.savefig(size_path)
