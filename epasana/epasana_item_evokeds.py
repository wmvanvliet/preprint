"""
Create epochs where each epoch is the evoked to a single item.
"""
import mne
import pandas as pd
import numpy as np
from tqdm import tqdm

# Shh
mne.set_log_level('ERROR')

# Path to the epasana dataset on scratch
data_path = '/m/nbe/scratch/epasana'

# Read epochs and create the stimulus order. Drop epochs that do no have proper metadata.
stimuli = pd.read_csv('stimulus_selection.csv', index_col=0)

all_epochs = []
for subject in tqdm(range(1, 16)):
    epochs = mne.read_epochs(f'{data_path}/bids/derivatives/marijn/sub-{subject:02d}/meg/sub-{subject:02d}-epo.fif')
    epochs.crop(0, 0.8).resample(100).pick_types(meg='grad')
    epochs.metadata = epochs.metadata.join(stimuli['y'], on='tif_file')
    epochs.drop(np.flatnonzero(np.isnan(epochs.metadata.y).values))
    all_epochs.append(epochs)
epochs = mne.concatenate_epochs(all_epochs)
assert len(epochs) == (560 * 15)

# Create new epochs by averaging all repetitions of each item
data = []
for _, item in tqdm(stimuli.iterrows(), total=len(stimuli)):
    data.append(epochs[f'y == {item.y}'].average().data)
epochs_items = mne.EpochsArray(data, epochs.info, event_id=dict(item=1),
                               tmin=epochs.times[0])
epochs_items.metadata = stimuli
epochs_items.save(f'{data_path}/bids/derivatives/reading_models/items-epo.fif')
