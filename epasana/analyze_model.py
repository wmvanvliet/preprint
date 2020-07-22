#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 23:10:47 2020

@author: shimizt1
"""

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

def analyze_model(model_path, data_path='/m/nbe/scratch/epasana',
                  meta_path = '/m/nbe/scratch/epasana/bids/stimuli.csv',
                  feature_layers=[5, 12, 22, 32, 42],
                  classifier_layers=[1, 4, 8],
                  noise_path='noise.pdf', rot_path='rotation.pdf',
                  len_path='len.pdf', rsa_path='rsa_analysis.pdf'):
    """Analyze model on how it responds to noises, rotations and word lengths,
    and save results in image files.
    
    Parameters
    ----------
    model_path: str
        Path to the VGGSem model.
    data_path: str
        Path to the epasana dataset folder.
    meta_path: str
        Path to the csv file containing meta data of epasana images.
    feature_layers: list of int
        Index of feature layers to use for analysis.
    classifier_layers: list of int
        Index of classifier layers to use for analysis.
    noise_path: str
        File name to save the noise DSMs plot as.
    rot_path: str
        File name to save the rotation DSMs plot as.
    len_path: str
        File name to save the word length DSMs plot as.
    rsa_path: str
        File name to save the RSA analysis plot as.
    """
    
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
    model = networks.vgg_sem.from_checkpoint(checkpoint)
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        outputs = model.get_layer_activations(img_noise,
                                              feature_layers=feature_layers,
                                              classifier_layers=classifier_layers)
    
    # noisy image DSM
    DSM_pixelwise_noise = mne_rsa.compute_dsm(img_noise, metric='euclidean')
    
    # calculate RSA between model DSMs and image DSM
    n_loops = len(feature_layers) + len(classifier_layers) + 1
    rsa_noise = []
    for i in range(n_loops):
        dsm = mne_rsa.compute_dsm(next(outputs), metric='correlation')
        rsa_noise.append(mne_rsa.rsa(dsm, DSM_pixelwise_noise))
        del dsm
    
    # get model activations
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        outputs = model.get_layer_activations(img_noise,
                                              feature_layers=feature_layers,
                                              classifier_layers=classifier_layers)
    
    # plot model DSMs
    h = int(np.ceil(np.sqrt(n_loops)))
    w = int(np.ceil(n_loops/h))
    figsize = (int(w*5), int(h*5))
    fig, axes = plt.subplots(h, w, figsize=figsize)
    for i in range(n_loops):
        dsm = mne_rsa.compute_dsm(next(outputs), metric='correlation')
        ax = axes[i // w, i % w]
        sns.heatmap(distance.squareform(dsm), ax=ax, square=True)
        del dsm
    
    rsa_noise_round = np.round(rsa_noise, 2)    
    
    feature = ("Feature Layer " + pd.Series(feature_layers, dtype=str)
               + " (RSA=" + pd.Series(rsa_noise_round[:len(feature_layers)], dtype=str) + ")")
    classifier = ("Classifier Layer " + pd.Series(classifier_layers, dtype=str)
                 + " (RSA=" + pd.Series(rsa_noise_round[len(feature_layers):-1], dtype=str) + ")")
    sem = "Semantics Layer (RSA=" + str(rsa_noise_round[-1])  + ")"
    
    for i, f in enumerate(feature):
        axes[i // w, i % w].title.set_text(f)
        
    for i, c in enumerate(classifier):
        j = i + len(feature)
        axes[j // w, j % w].title.set_text(c)
        
    axes[-1, (n_loops - 1) % w].title.set_text(sem)
    
    n_empty = (w * h) - n_loops
    for i in range(1, n_empty + 1):
        fig.delaxes(axes[-1, -i])
    
    # save model DSMs for noisy imput
    fig.savefig(noise_path)
    
    def create_image_rotation(data_path=data_path, index=300, 
                              angles=[0, 45, -45, 90, -90, 135, -135, 180]):
        """Get the stimulus images presented during the MEG experiment with rotation.
        Parameters
        ----------
        data_path : str
            Path to the epasana dataset on scratch.
        index: int
            index of an image to generate noisy images based on
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
    
    # create images with different rotations
    img_rotation = create_image_rotation()
    
    # get image DSM
    DSM_pixelwise_rot = mne_rsa.compute_dsm(img_rotation, metric='euclidean')
    
    # Load the model and feed through the images
    checkpoint = torch.load(model_path, map_location='cpu')
    model = networks.vgg_sem.from_checkpoint(checkpoint)
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        outputs = model.get_layer_activations(img_rotation,
                                              feature_layers=feature_layers,
                                              classifier_layers=classifier_layers)
    
    # get RSAs between model DSMs and image DSM
    rsa_rot = []
    for i in range(n_loops):
        dsm = mne_rsa.compute_dsm(next(outputs), metric='correlation')
        rsa_rot.append(mne_rsa.rsa(dsm, DSM_pixelwise_rot))
        del dsm
    
    # get model activations
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        outputs = model.get_layer_activations(img_rotation,
                                              feature_layers=feature_layers,
                                              classifier_layers=classifier_layers)

    # plot model DSMs
    fig, axes = plt.subplots(h, w, figsize=figsize)
    for i in range(n_loops):
        dsm = mne_rsa.compute_dsm(next(outputs), metric='correlation')
        ax = axes[i // w, i % w]
        sns.heatmap(distance.squareform(dsm), ax=ax, square=True)
        del dsm
    
    rsa_rot_round = np.round(rsa_rot, 2)    
    
    feature = ("Feature Layer " + pd.Series(feature_layers, dtype=str)
               + " (RSA=" + pd.Series(rsa_rot_round[:len(feature_layers)], dtype=str) + ")")
    classifier = ("Classifier Layer " + pd.Series(classifier_layers, dtype=str)
                 + " (RSA=" + pd.Series(rsa_rot_round[len(feature_layers):-1], dtype=str) + ")")
    sem = "Semantics Layer (RSA=" + str(rsa_rot_round[-1])  + ")"
    
    for i, f in enumerate(feature):
        axes[i // w, i % w].title.set_text(f)
        
    for i, c in enumerate(classifier):
        j = i + len(feature)
        axes[j // w, j % w].title.set_text(c)
    
    axes[-1, (n_loops - 1) % w].title.set_text(sem)
    
    for i in range(1, n_empty + 1):
        fig.delaxes(axes[-1, -i])
    
    # save model DSMs image for rotated input
    fig.savefig(rot_path)
    
    meta = pd.read_csv(meta_path)
    lengths = np.array([len(word) for word in meta[meta.text.notna()].text])
    unique_len = np.unique(lengths)
    
    # pseudoword is chosen since other types do not have a word with 9 letters
    # 'words' contain indices of 3 pseudowords with different lengths (7, 8, 9)
    words = []
    for l in unique_len:
        words_l = meta[meta.text.notna()][lengths == l]
        words.append(words_l[words_l.type=='pseudoword'].iloc[0].name)
        
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
    
    # create images with different word lengths
    img_len = create_image_len(index=words)
    
    # Load the model and feed through the images
    checkpoint = torch.load(model_path, map_location='cpu')
    model = networks.vgg_sem.from_checkpoint(checkpoint)
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        outputs = model.get_layer_activations(img_len,
                                              feature_layers=feature_layers,
                                              classifier_layers=classifier_layers)
    
    # get image DSM
    DSM_pixelwise_len = mne_rsa.compute_dsm(img_len, metric='euclidean')
    
    # get RSAs between the image DSM and model DSMs
    rsa_len = []
    for i in range(n_loops):
        dsm = mne_rsa.compute_dsm(next(outputs), metric='correlation')
        rsa_len.append(mne_rsa.rsa(dsm, DSM_pixelwise_len))
        del dsm
    
    # get model activations
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        outputs = model.get_layer_activations(img_len,
                                              feature_layers=feature_layers,
                                              classifier_layers=classifier_layers)
    
    # plot model DSMs for different word lengths
    fig, axes = plt.subplots(h, w, figsize=figsize)
    for i in range(n_loops):
        dsm = mne_rsa.compute_dsm(next(outputs), metric='correlation')
        ax = axes[i // w, i % w]
        sns.heatmap(distance.squareform(dsm), ax=ax, square=True)
        del dsm
    
    rsa_len_round = np.round(rsa_len, 2).astype(str)
    
    feature = ("Feature Layer " + pd.Series(feature_layers, dtype=str)
               + " (RSA=" + pd.Series(rsa_len_round[:len(feature_layers)]) + ")")
    classifier = ("Classifier Layer " + pd.Series(classifier_layers, dtype=str)
                 + " (RSA=" + pd.Series(rsa_len_round[len(feature_layers):-1]) + ")")
    sem = "Semantics Layer (RSA=" + str(rsa_len_round[-1]) + ")"
    
    for i, f in enumerate(feature):
        axes[i // w, i % w].title.set_text(f)
        
    for i, c in enumerate(classifier):
        j = i + len(feature)
        axes[j // w, j % w].title.set_text(c)
        
    axes[-1, (n_loops - 1) % w].title.set_text(sem)
    
    for i in range(1, n_empty + 1):
        fig.delaxes(axes[-1, -i])
    
    # save DSMs image for different word lengths
    fig.savefig(len_path)
    
    # plot RSAs bar graph
    conv_layers = 'conv' + pd.Series(feature_layers, dtype=str)
    fc_layers = 'fc' + pd.Series(classifier_layers, dtype=str)
    layers = pd.concat([conv_layers, fc_layers, pd.Series(['sem'])], ignore_index=True)
    rsas = [rsa_noise, rsa_rot, rsa_len]
    
    n_plots = 3
    fig, axes = plt.subplots(n_plots, sharex=True, sharey=True, figsize=(6, 8))
    titles = ["Noise", "Rotation", "Word Length"]
    fig.suptitle("RSA analysis")
    
    for i in range(n_plots):
        axes[i].bar(x=layers, height=rsas[i])
        axes[i].set_title(titles[i], fontdict={'verticalalignment': 'center'}, rotation='vertical', x=-0.125, y=0.5)
    plt.setp(axes, ylim=(-0.6, 1.2));
    plt.savefig("rsa_path")
    