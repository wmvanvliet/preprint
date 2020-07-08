"""
Utility functions for grabbing pilot data.
"""
import mne
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd


def get_stimulus_images(stimuli, data_path='/m/nbe/scratch/epasana'):
    """Get the stimulus images presented during the MEG experiment

    Parameters
    ----------
    stimuli : DataFrame
        Stimulus information
    data_path : str
        Path to the epasana dataset on scratch.

    Returns
    -------
    images : tensor, shape (n_images, n_colors, width, height)
        A tensor containing the bitmap data of the images presented to the
        subject during the MEG experiment. Ready to feed into a model.
    """
    # Transform the images to a 60x60 pixel image suitable for feeding through
    # the model. This is the same transform as used in train_net.py during the
    # training of the model.
    preproc = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load the PNG images presented in the MEG experiment and apply the
    # transformation to them. Also keep track of the filesizes, as this is a decent
    # measure of visual complexity and perhaps useful in the RSA analysis.
    images = []
    for fname in tqdm(stimuli['tif_file'], desc='Reading images'):
        with Image.open(f'{data_path}/bids/stimulus_images/{fname}') as orig:
            image = Image.new('RGB', (224, 224), '#696969')
            image.paste(orig, (12, 62))
            image = preproc(image).unsqueeze(0)
            images.append(image)
    return torch.cat(images, 0)
