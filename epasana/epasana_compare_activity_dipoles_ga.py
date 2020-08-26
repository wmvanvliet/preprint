import pandas as pd
from matplotlib import pyplot as plt

from config import fname

dip_layer_corr = []
bad_subjects = {3, 6, 13, 15}
for subject in range(1, 16):
    #if subject in bad_subjects:
    #     continue
    df = pd.read_csv(fname.dip_layer_corr(subject=subject))
    df['subject'] = subject
    dip_layer_corr.append(df)
dip_layer_corr = pd.concat(dip_layer_corr, ignore_index=True)

corr_mean = dip_layer_corr.groupby('group').agg('mean').iloc[:, 2:-1]
corr_sem = dip_layer_corr.groupby('group').agg('sem').iloc[:, 2:-1]

##
_, axes = plt.subplots(2, 4, sharex=True, figsize=(8, 5))
for region, ax in zip(corr_mean.index, axes.flat):
    ax.bar(corr_mean.columns, corr_mean.loc[region], yerr=corr_sem.loc[region])
    ax.set_title(region)
plt.tight_layout()


##
import mne
import numpy as np
from tqdm import tqdm
from scipy.stats import zscore
metadata = []
dip_timecourses = []
dip_selection = []
for subject in tqdm(range(1, 16)):
    epochs = mne.read_epochs(fname.epochs(subject=subject), preload=False)
    dip_t = np.load(fname.dip_timecourses(subject=subject))['proj']
    dip_t = dip_t[:, np.argsort(epochs.metadata.tif_file), :]
    dip_timecourses.append(dip_t)
    dip_sel = pd.read_csv(fname.dip_selection(subject=subject), sep='\t')
    dip_sel['subject'] = subject
    dip_selection.append(dip_sel)
    m = epochs.metadata.copy()
    m['subject'] = subject
    m['epoch_index'] = np.arange(len(epochs))
    metadata.append(m)
metadata = pd.concat(metadata, ignore_index=True)
dip_selection = pd.concat(dip_selection, ignore_index=True)

stimuli = pd.read_csv(fname.stimulus_selection)
stimuli['type'] = stimuli['type'].astype('category')

##
time_rois = {
    'LeftOcci1': slice(*np.searchsorted(epochs.times, [0.065, 0.115])),
    'RightOcci1': slice(*np.searchsorted(epochs.times, [0.065, 0.115])),
    'LeftOcciTemp2': slice(*np.searchsorted(epochs.times, [0.14, 0.2])),
    'RightOcciTemp2': slice(*np.searchsorted(epochs.times, [0.185, 0.220])),
    'LeftTemp3': slice(*np.searchsorted(epochs.times, [0.300, 0.400])),
    'RightTemp3': slice(*np.searchsorted(epochs.times, [0.300, 0.400])),
    'LeftFront2-3': slice(*np.searchsorted(epochs.times, [0.300, 0.500])),
    'RightFront2-3': slice(*np.searchsorted(epochs.times, [0.300, 0.500])),
    'LeftPar2-3': slice(*np.searchsorted(epochs.times, [0.250, 0.350])),
}

mean_acts = []
fig, axes = plt.subplots(2, 3, sharex=True, figsize=(10, 10))
groups =['LeftOcci1', 'LeftOcciTemp2', 'LeftTemp3', 'RightOcci1', 'RightOcciTemp2', 'RightTemp3']
for group, ax in zip(groups, axes.flat):
    group_act = []
    sel = dip_selection.query(f'group=="{group}"')
    for sub, dip, neg in zip(sel.subject, sel.dipole, sel.neg):
        print(sub, dip, neg)
        tc = dip_timecourses[sub - 1][dip] * (-1 if neg else 1)
        quant = zscore(tc[:, time_rois[group]].mean(axis=1))
        df = metadata.query(f'subject=={sub}').sort_values('tif_file')
        df[group] = quant
        group_act.append(df)
    group_act = pd.concat(group_act)

    mean_act = group_act.groupby('tif_file').agg('mean')[group]
    mean_acts.append(mean_act)
    cat = group_act.groupby('tif_file').agg('first').type
    #ax.plot(mean_act.values, linewidth=1)
    #ax.scatter(np.arange(len(mean_act)), mean_act.values, s=2)
    x_offset = 0
    for j, cat in enumerate(stimuli['type'].unique()):
        cat_index = np.flatnonzero(stimuli['type'] == cat)
        selection = mean_act[cat_index]
        ax.scatter(np.arange(len(selection)) + x_offset, selection, s=2)
        x_offset += len(selection)
        ax.plot(cat_index[[0, -1]], selection.mean().repeat(2), color=plt.get_cmap('tab10').colors[j], label=cat)
    ax.set_title(group)
    if group == 'LeftOcci1':
        ax.legend()
plt.tight_layout()
mean_acts = np.array(mean_acts)

##
# Load model layer activations
import pickle
from scipy.stats import rankdata
model_name = 'vgg11_first_imagenet_then_epasana-10kwords_epasana-nontext'
with open(fname.model_dsms(name=model_name), 'rb') as f:
    d = pickle.load(f)
layer_activity = np.array(d['layer_activity'])[:, stimuli.index]
layer_names = d['dsm_names'][:-4]

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

r1 = pearsonr(mean_acts, layer_activity[1::2])
r2 = spearmanr(mean_acts, layer_activity[1::2])
fig, axes = plt.subplots(2, 3, sharex=True)
for r, ax, group in zip(r1, axes.flat, groups):
    ax.bar(np.arange(len(r)), r)
    ax.set_title(group)
