import rsa
import mne
import pandas as pd
import numpy as np
import pickle

epochs = mne.read_epochs('data/pilot_data/pilot2/pilot2_epo.fif')
epochs = epochs[['word', 'symbols', 'consonants']]
epochs.crop(0, 0.8).resample(100).pick_types(meg='grad')
stimuli = epochs.metadata.groupby('text').agg('first').sort_values('event_id')
stimuli['y'] = np.arange(len(stimuli))
metadata = pd.merge(epochs.metadata, stimuli[['y']], left_on='text', right_index=True).sort_index()
assert np.array_equal(metadata.event_id.values.astype(int), epochs.events[:, 2])

model_name = 'vgg_first_imagenet64_then_tiny-words_tiny-imagenet'
with open(f'../data/dsms/pilot2_{model_name}_dsms.pkl', 'rb') as f:
    dsm_models = pickle.load(f)
    dsms = dsm_models['dsms']
    dsm_names = dsm_models['dsm_names']

rsa_results = rsa.rsa_epochs(
    epochs,
    dsms,
    y=metadata['y'],
    spatial_radius=0.04,
    temporal_radius=0.05,
    epochs_dsm_metric='correlation',
    verbose=True,
    n_jobs=8,
)

for r, name in zip(rsa_results, dsm_names):
    r.comment = name

mne.write_evokeds('../data/pilot_data/pilot2/pilot2_rsa_results-ave.fif', rsa_results)
