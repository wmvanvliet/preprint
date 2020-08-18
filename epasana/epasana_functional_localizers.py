"""
Create functional localizers for the three landmarks: visual, letter string and m400.
Quantify activity in each landmark.
"""
import sys
import mne
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from config import fname, subjects

if len(sys.argv) > 1:
    subject = int(sys.argv[1])
else:
    subject = 'ga'

##
if subject == 'ga':
    epochs = mne.read_epochs(fname.ga_epochs)
else:
    epochs = mne.read_epochs(fname.epochs(subject=subject))

info_102 = mne.io.read_info(fname.info_102)
info_102['sfreq'] = epochs.info['sfreq']

# Combine gradiometer pairs
epochs.pick_types(meg='grad')
grads_comb = np.linalg.norm(epochs.get_data().reshape(len(epochs), 102, 2, len(epochs.times)), axis=2)

## Make t-maps with the proper contrasts to hunt for the ROIs
# Contrast between noisy and non-noisy stimuli.
cond1 = grads_comb[epochs.metadata.type == 'noisy word']
cond2 = grads_comb[epochs.metadata.type != 'noisy word']
ts_noise, ps_noise = ttest_ind(cond1, cond2, axis=0)
tmap_noise = mne.EvokedArray(ts_noise.repeat(2, axis=0), epochs.info, tmin=epochs.tmin, comment='noise contrast t-values')

# Contrast between letter and non-letter stimuli.
cond1 = grads_comb[epochs.metadata.type.isin(['word', 'pseudoword', 'consonants'])]
cond2 = grads_comb[epochs.metadata.type == 'symbols']
ts_letter, ps_letter = ttest_ind(cond1, cond2, axis=0)
tmap_letter = mne.EvokedArray(ts_letter.repeat(2, axis=0), epochs.info, tmin=epochs.tmin, comment='letter contrast t-values')

# Contrast between (pseudo-)word and non-word stimuli.
cond1 = grads_comb[epochs.metadata.type.isin(['word', 'pseudoword'])]
cond2 = grads_comb[epochs.metadata.type.isin(['symbols', 'consonants'])]
ts_word, ps_word = ttest_ind(cond1, cond2, axis=0)
tmap_word = mne.EvokedArray(ts_word.repeat(2, axis=0), epochs.info, tmin=epochs.tmin, comment='word contrast t-values')

## Plot tvalues and overlay manually chosen temporal ROIs
ROI_noise_temp = (0.065, 0.135)  # Manually chosen values
ROI_letter_temp = (0.16, 0.22)
ROI_word_temp = (0.27, 0.54)
ROI_noise_temp_idx = slice(*np.searchsorted(epochs.times, ROI_noise_temp))
ROI_letter_temp_idx = slice(*np.searchsorted(epochs.times, ROI_letter_temp))
ROI_word_temp_idx = slice(*np.searchsorted(epochs.times, ROI_word_temp))

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
axes[0].plot(tmap_noise.times, ts_noise.T, color='black', alpha=0.2)
axes[0].set_ylabel('t-value')
axes[0].set_title(tmap_noise.comment)
axes[0].axvspan(*ROI_noise_temp, alpha=0.2, color='C0', label='Early visual')
axes[0].legend(loc='upper right')
axes[1].plot(tmap_letter.times, ts_letter.T, color='black', alpha=0.2)
axes[1].set_ylabel('t-value')
axes[1].set_title(tmap_letter.comment)
axes[1].axvspan(*ROI_letter_temp, alpha=0.2, color='C1', label='Letter string')
axes[1].legend(loc='upper right')
axes[2].plot(tmap_word.times, ts_word.T, color='black', alpha=0.2)
axes[2].set_ylabel('t-value')
axes[2].set_xlabel('Time (s)')
axes[2].axvspan(*ROI_word_temp, alpha=0.2, color='C2', label='m400')
axes[2].legend(loc='upper right')
axes[2].set_title(tmap_word.comment)
plt.tight_layout()

## Plot tvalues as topomaps and overlay manually chosen spatial ROIs
ROI_noise_spat = np.flatnonzero(ts_noise[:, ROI_noise_temp_idx].mean(axis=1).repeat(2, axis=0) > 8)
ROI_letter_spat = np.flatnonzero(ts_letter[:, ROI_letter_temp_idx].mean(axis=1).repeat(2, axis=0) > 3)
ROI_word_spat = np.flatnonzero(ts_word[:, ROI_word_temp_idx].mean(axis=1).repeat(2, axis=0) > 3)

cond1 = np.flatnonzero(epochs.metadata.type == 'noisy word')
cond2 = np.flatnonzero(epochs.metadata.type != 'noisy word')
contrast1 = mne.combine_evoked((epochs[cond1].average(), -epochs[cond2].average()), weights='equal')
mask = np.zeros(contrast1.data.shape, np.bool)
mask[ROI_noise_spat, ROI_noise_temp_idx] = True
contrast1.plot_joint([0.1, 0.17, 0.4],
                     ts_args=dict(spatial_colors=False),
                     topomap_args=dict(mask=mask, mask_params=dict(markersize=8)),
                     title='noise contrast')

cond1 = np.flatnonzero(epochs.metadata.type.isin(['word', 'pseudoword', 'consonants']))
cond2 = np.flatnonzero(epochs.metadata.type == 'symbols')
contrast2 = mne.combine_evoked((epochs[cond1].average(), -epochs[cond2].average()), weights='equal')
mask = np.zeros(contrast2.data.shape, np.bool)
mask[ROI_letter_spat, ROI_letter_temp_idx] = True
contrast2.plot_joint([0.1, 0.17, 0.4],
                     ts_args=dict(spatial_colors=False),
                     topomap_args=dict(mask=mask, mask_params=dict(markersize=8)),
                     title='letter contrast')

cond1 = np.flatnonzero(epochs.metadata.type.isin(['word', 'pseudoword']))
cond2 = np.flatnonzero(epochs.metadata.type == 'consonants')
contrast3 = mne.combine_evoked((epochs[cond1].average(), -epochs[cond2].average()), weights='equal')
mask = np.zeros(contrast3.data.shape, np.bool)
mask[ROI_word_spat, ROI_word_temp_idx] = True
contrast3.plot_joint([0.1, 0.17, 0.4],
                     ts_args=dict(spatial_colors=False),
                     topomap_args=dict(mask=mask, mask_params=dict(markersize=8)),
                     title='word contrast')

if subject == 'ga':
    mne.write_evokeds(fname.ga_contrasts, [contrast1, contrast2, contrast3])
else:
    mne.write_evokeds(fname.contrasts(subject=subject), [contrast1, contrast2, contrast3])
