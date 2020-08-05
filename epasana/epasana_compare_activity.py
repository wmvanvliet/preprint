import sys
import mne
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import zscore, rankdata

if len(sys.argv) > 1:
    subject = int(sys.argv[1])
else:
    subject = 'ga'

stimuli = pd.read_csv('stimulus_selection.csv')
stimuli['type'] = stimuli['type'].astype('category')
info = mne.io.read_info('info_102.fif')

# Read MEG data
if subject == 'ga':
    epochs = mne.read_epochs('../data/epasana/items-epo.fif')
    epochs.metadata['tif_file'] = stimuli['tif_file'].values
else:
    epochs = mne.read_epochs(f'm:/scratch/epasana/bids/derivatives/marijn/sub-{subject:02d}/meg/sub-{subject:02d}-epo.fif')
epochs.pick_types(meg='grad')

# Drop any stimuli for which there are no corresponding epochs
stimuli = stimuli[stimuli.tif_file.isin(epochs.metadata.tif_file)]

# Re-order epochs to be in the order of the stimuli
epochs.metadata['epoch_index'] = np.arange(len(epochs))
epochs = epochs[epochs.metadata.sort_values('tif_file').epoch_index.values]

# Create evoked
evokeds = []
for t in stimuli.type.unique():
    ev = epochs[f'type=="{t}"'].average()
    #grads_comb = np.linalg.norm(ev.data.reshape(2, 102, 80), axis=0)
    #ev = mne.EvokedArray(grads_comb, info, tmin=ev.times[0])
    ev.comment = t
    evokeds.append(ev)

# Load model layer activations
model_name = 'vgg11_first_imagenet_then_epasana-10kwords_epasana-nontext'
with open(f'../data/dsms/epasana_{model_name}_dsms.pkl', 'rb') as f:
    d = pickle.load(f)
layer_activity = np.array(d['layer_activity'])[:, stimuli.index]
layer_names = d['dsm_names'][:-4]

# # Plot layer activations
# fig, axes = plt.subplots(4, 4, sharex=True, figsize=(16, 10))
# for i, ax in enumerate(axes.ravel()):
#     ax.plot(layer_activity[i])
#     ax.set_title(layer_names[i])
#     for j, cat in enumerate(stimuli['type'].unique()):
#         cat_index = np.flatnonzero(stimuli['type'] == cat)
#         ax.plot(cat_index[[0, -1]], layer_activity[i][cat_index].mean().repeat(2), color=matplotlib.cm.magma(j / 4), label=cat)
#     if i == 0:
#         ax.legend(loc='upper right')
# plt.tight_layout()

def pearsonr(x, y, *, x_axis=-1, y_axis=-1):
    """Compute Pearson correlation along the given axis."""
    x = zscore(x, axis=x_axis)
    y = zscore(y, axis=y_axis)
    return np.tensordot(x, y, (x_axis, y_axis)) / x.shape[x_axis]

def spearmanr(x, y, *, x_axis=-1, y_axis=-1):
    """Compute Spearman rank correlation along the given axis."""
    x = rankdata(x, axis=x_axis)
    y = rankdata(y, axis=y_axis)
    return pearsonr(x, y, x_axis=x_axis, y_axis=y_axis)

info['sfreq'] = epochs.info['sfreq']
evokeds_r = []
# # Combine grads
# grads_comb = np.linalg.norm(epochs._data.reshape(588, 2, 102, 80), axis=1)
# for r, name in zip(pearsonr(layer_activity, grads_comb, x_axis=1, y_axis=0), layer_names):
#     ev = mne.EvokedArray(r.repeat(2, axis=0), epochs.info, tmin=epochs.times[0], comment=name)
#     evokeds_r.append(ev)
for r, name in zip(spearmanr(layer_activity, epochs._data, x_axis=1, y_axis=0), layer_names):
    ev = mne.EvokedArray(r, epochs.info, tmin=epochs.times[0], comment=name)
    evokeds_r.append(ev)
# mne.viz.plot_evoked_topo([evokeds_r[i] for i in [3, 9, 13, 15]], scalings=dict(grad=1), ylim=dict(grad=[0, 0.5]), merge_grads=False)

if subject == 'ga':
    mne.write_evokeds('m:/scratch/reading_models/epasana/ga_layer_corr-ave.fif', evokeds_r)
else:
    mne.write_evokeds(f'm:/scratch/reading_models/epasana/sub-{subject:02}_layer_corr-ave.fif', evokeds_r)
