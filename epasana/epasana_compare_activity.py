import mne
import mne_rsa
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import zscore, spearmanr
from tqdm import tqdm

epochs = mne.read_epochs('../data/epasana/items-epo.fif')
stimuli = pd.read_csv('stimulus_selection.csv')
stimuli['type'] = stimuli['type'].astype('category')

ga = []
for t in stimuli.type.unique():
    ev = epochs[f'type=="{t}"'].average()
    ev.comment = t
    ga.append(ev)


model_name = 'vgg11_first_imagenet_then_epasana-10kwords_epasana-nontext'
with open(f'../data/dsms/epasana_{model_name}_dsms.pkl', 'rb') as f:
    d = pickle.load(f)
layer_activity = d['layer_activity'][1::2]
layer_names = d['dsm_names'][1:-4:2]

dsms = [mne_rsa.compute_dsm(a, metric='euclidean') for a in layer_activity]
mne_rsa.plot_dsms(dsms, layer_names, n_rows=4)

fig, axes = plt.subplots(4, 2, sharex=True, figsize=(16, 10))
for i, ax in enumerate(axes.ravel()):
    ax.plot(layer_activity[i])
    ax.set_title(layer_names[i])
    for j, cat in enumerate(stimuli['type'].unique()):
        cat_index = np.flatnonzero(stimuli['type'] == cat)
        ax.plot(cat_index[[0, -1]], layer_activity[i][cat_index].mean().repeat(2), color=matplotlib.cm.magma(j / 4), label=cat)
    if i == 0:
        ax.legend(loc='upper right')
plt.tight_layout()

epochs_norm = epochs.copy()
epochs_norm._data = zscore(epochs_norm._data, axis=0)

evokeds = []
for i, name in enumerate(layer_names):
    if name.endswith('relu'):
        continue
    design_matrix = np.hstack([np.ones((len(epochs_norm), 1)), zscore(layer_activity[i])[:, None]])
    r = mne.stats.linear_regression(epochs_norm, design_matrix, ['offset', name])
    ev = r[name].beta
    ev.comment = name
    evokeds.append(ev)

# HACK!
data = np.zeros(epochs._data.shape[1:])
for ch in tqdm(range(len(epochs.ch_names))):
    for t in range(len(epochs.times)):
        data[ch, t] = spearmanr(layer_activity[2], epochs._data[:, ch, t])[0]
ev1 = mne.EvokedArray(abs(data), epochs.info, tmin=epochs.times[0], comment=layer_names[2])

data = np.zeros(epochs._data.shape[1:])
for ch in tqdm(range(len(epochs.ch_names))):
    for t in range(len(epochs.times)):
        data[ch, t] = spearmanr(layer_activity[8], epochs._data[:, ch, t])[0]
ev2 = mne.EvokedArray(abs(data), epochs.info, tmin=epochs.times[0], comment=layer_names[8])

data = np.zeros(epochs._data.shape[1:])
for ch in tqdm(range(len(epochs.ch_names))):
    for t in range(len(epochs.times)):
        data[ch, t] = spearmanr(layer_activity[12], epochs._data[:, ch, t])[0]
ev3 = mne.EvokedArray(abs(data), epochs.info, tmin=epochs.times[0], comment=layer_names[12])

data = np.zeros(epochs._data.shape[1:])
for ch in tqdm(range(len(epochs.ch_names))):
    for t in range(len(epochs.times)):
        data[ch, t] = spearmanr(layer_activity[14], epochs._data[:, ch, t])[0]
ev4 = mne.EvokedArray(abs(data), epochs.info, tmin=epochs.times[0], comment=layer_names[14])
mne.viz.plot_evoked_topo([ev1, ev2, ev3, ev4], merge_grads=True, scalings=dict(grad=1), ylim=dict(grad=[0, 0.5]))
