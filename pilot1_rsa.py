import rsa
import mne
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance

epochs = mne.read_epochs('data/pilot_data/pilot1/pilot1_epo.fif')
epochs = epochs[['word', 'symbols', 'consonants']]
epochs.crop(0, 0.8).resample(100).pick_types(meg='grad')
epochs.metadata.columns = ['text']
event_id = {10: 'word', 20: 'consonants', 30: 'symbols'}
epochs.metadata['type'] = [event_id[c] for c in epochs.events[:, 2]]
epochs.metadata['event_id'] = epochs.events[:, 2].astype(int)
stimuli = epochs.metadata.groupby('text').agg('first').sort_values('event_id')
stimuli['y'] = np.arange(len(stimuli))
metadata = pd.merge(epochs.metadata, stimuli[['y']], left_on='text', right_index=True).sort_index()
assert np.array_equal(metadata.event_id.values.astype(int), epochs.events[:, 2])

def words_only(x, y):
    if x != 'word' or y != 'word':
        return 1
    else:
        return 0

def letters_only(x, y):
    if x == 'symbols' or y == 'symbols':
        return 1
    else:
        return 0

words_only_dsm = rsa.compute_dsm(stimuli[['type']], metric=words_only)
letters_only_dsm = rsa.compute_dsm(stimuli[['type']], metric=letters_only)

plt.matshow(distance.squareform(words_only_dsm))
plt.title('words_only_dsm')
plt.matshow(distance.squareform(letters_only_dsm))
plt.title('letters_only_dsm')

rsa_results = rsa.rsa_epochs(
    epochs,
    [words_only_dsm, letters_only_dsm],
    y=metadata['y'],
    epochs_dsm_metric='correlation',
    rsa_metric='spearman',
    verbose=True,
    spatial_radius=0.02,
    temporal_radius=0.02,
    n_jobs=4,
)
rsa_results[0].comment = 'words only'
rsa_results[1].comment = 'letters only'

mne.write_evokeds('data/pilot_data/pilot1/pilot1_rsa_results-ave.fif', rsa_results)
