import mne
import mne_rsa
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import zscore

epochs = mne.read_epochs('../data/epasana/items-epo.fif')
stimuli = pd.read_csv('stimulus_selection.csv')
stimuli['type'] = stimuli['type'].astype('category')

model_name = 'vgg11_first_imagenet_then_epasana-10kwords_epasana-nontext'
with open(f'../data/dsms/epasana_{model_name}_dsms.pkl', 'rb') as f:
    d = pickle.load(f)
layer_activity = d['layer_activity']
layer_names = d['dsm_names'][:-4]

dsms = [mne_rsa.compute_dsm(a, metric='euclidean') for a in layer_activity]
mne_rsa.plot_dsms(dsms, layer_names, n_rows=4)

fig, axes = plt.subplots(4, 4, sharex=True, figsize=(16, 10))
for i, ax in enumerate(axes.ravel()):
    ax.plot(layer_activity[i])
    ax.set_title(layer_names[i])
    for j, cat in enumerate(stimuli['type'].unique()):
        cat_index = np.flatnonzero(stimuli['type'] == cat)
        ax.plot(cat_index[[0, -1]], layer_activity[i][cat_index].mean().repeat(2), color=matplotlib.cm.magma(j / 4))
plt.tight_layout()

no_noise_idx = stimuli.query('type != "noisy word"').index.values
epochs_norm = epochs[no_noise_idx]
epochs_norm._data = zscore(epochs_norm._data, axis=0)

evokeds = []
for i, name in enumerate(layer_names):
    if name.endswith('relu'):
        continue
    design_matrix = np.hstack([np.ones((len(epochs_norm), 1)), zscore(layer_activity[i])[no_noise_idx, None]])
    r = mne.stats.linear_regression(epochs_norm, design_matrix, ['offset', name])
    ev = r[name].beta
    ev.comment = name
    evokeds.append(ev)
