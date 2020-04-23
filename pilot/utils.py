"""
Utility functions for grabbing pilot data.
"""
import mne
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


def get_stimulus_info(subject=2, data_path='/m/nbe/scratch/reading_models'):
    """Get information about the stimuli presented during the MEG experiment

    Parameters
    ----------
    subject : 1 | 2
        The pilot subject to read the stimuli for. Defaults to 2.

    Returns
    -------
    stimuli : pandas.DataFrame
        A table with information about the stimuli presented to the subject.
    """
    epochs = mne.read_epochs(f'{data_path}/pilot_data/pilot{subject:d}/pilot{subject:d}_epo.fif', preload=False)
    epochs = mne.read_epochs(f'{data_path}/pilot_data/pilot{subject:d}/pilot{subject:d}_epo.fif', preload=False)
    epochs = epochs[['word', 'symbols', 'consonants']]
    stimuli = epochs.metadata.groupby('text').agg('first').sort_values('event_id')
    stimuli['y'] = np.arange(len(stimuli))
    return stimuli


def get_stimulus_images(subject=2, stimuli=None, data_path='/m/nbe/scratch/reading_models'):
    """Get information about the stimuli presented during the MEG experiment

    Parameters
    ----------
    subject : 1 | 2
        The pilot subject to read the stimuli for. Defaults to 2.
    stimuli : DataFrame | None
        Information about the stimuli as returned by the `get_stimulus_info`
        function. When omitted, this is obtained automatically by calling this
        function.

    Returns
    -------
    images : tensor, shape (n_images, n_colors, width, height)
        A tensor containing the bitmap data of the images presented to the
        subject during the MEG experiment. Ready to feed into a model.
    """
    if stimuli is None:
        stimuli = get_stimulus_info(subject)

    # Transform the images to a 60x60 pixel image suitable for feeding through
    # the model. This is the same transform as used in train_net.py during the
    # training of the model.
    preproc = transforms.Compose([
        transforms.CenterCrop(208),
        transforms.Resize(64),
        transforms.CenterCrop(60),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load the PNG images presented in the MEG experiment and apply the
    # transformation to them. Also keep track of the filesizes, as this is a decent
    # measure of visual complexity and perhaps useful in the RSA analysis.
    images = []
    for fname in tqdm(stimuli['filename'], desc='Reading images'):
        with Image.open(f'{data_path}/pilot_data/pilot{subject:d}/stimuli/{fname}') as image:
            image = image.convert('RGB')
            image = preproc(image).unsqueeze(0)
            if image.shape[1] == 1:
                image = image.repeat(1, 3, 1, 1)
            images.append(image)
    return torch.cat(images, 0)
