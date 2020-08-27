import sys
import mne
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator
from scipy.stats import zscore, rankdata
from tqdm import tqdm
from joblib import Parallel, delayed

from config import fname

subject = int(sys.argv[1])

stimuli = pd.read_csv(fname.stimulus_selection)
stimuli['type'] = stimuli['type'].astype('category')
epochs = mne.read_epochs(fname.epochs(subject=subject), preload=False)
dip_timecourses = np.load(fname.dip_timecourses(subject=subject))['proj']
dip_selection = pd.read_csv(fname.dip_selection(subject=subject), sep='\t', index_col=0)

# Drop any stimuli for which there are no corresponding epochs
stimuli = stimuli[stimuli.tif_file.isin(epochs.metadata.tif_file)]

# Re-order epochs to be in the order of the stimuli
epochs.metadata['epoch_index'] = np.arange(len(epochs))
epochs = epochs[epochs.metadata.sort_values('tif_file').epoch_index.values]
dip_timecourses = dip_timecourses[:, epochs.metadata.epoch_index.values, :]

# Quantify dipole activity
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

dipole_act_quant = np.array([
    dip_timecourses[dip, :, time_rois[region]].mean(axis=1) * (-1 if neg else 1)
    for region, [dip, neg] in dip_selection.iterrows()
])

# Load model layer activations
model_name = 'vgg11_first_imagenet_then_epasana-10kwords_epasana-nontext'
with open(fname.model_dsms(name=model_name), 'rb') as f:
    d = pickle.load(f)
layer_activity = np.array(d['layer_activity'])[:, stimuli.index]
layer_names = d['dsm_names'][:-4]

stimuli.reset_index(drop=True)

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

r1 = pearsonr(dipole_act_quant, layer_activity[1::2])
r2 = spearmanr(dipole_act_quant, layer_activity[1::2])

dip_layer_corr = dip_selection.copy()
for r, name in zip(r1.T, layer_names[1::2]):
    dip_layer_corr[name] = r
dip_layer_corr.to_csv(fname.dip_layer_corr(subject=subject))

##
# Plot evoked dipole activations
time_slice = slice(*np.searchsorted(epochs.times, [-0.1, 0.6]))
n_cols = 2
n_rows = int(np.ceil(len(dip_selection) / 2))
fig1, axes = plt.subplots(n_rows, n_cols, sharex=True, figsize=(18, 10))
for (region, [dip, _]), ax in zip(dip_selection.iterrows(), axes.flat):
    for condition, cond_ind in stimuli.reset_index(drop=True).groupby('type').groups.items():
        act = dip_timecourses[dip, cond_ind, :].mean(axis=0)
        ax.plot(epochs.times[time_slice], act[time_slice], label=condition)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.legend(fontsize=8)
    ax.set_title(f'{region} (dip {dip})')
plt.tight_layout()

##
# Compare dipole activations with model activity
n_cols = 2
n_rows = int(np.ceil(len(dip_selection) / 2))
fig2, axes = plt.subplots(n_rows, n_cols, sharex=True, figsize=(1 + 2 * n_rows, 5))
for region, act, time_roi, ax in zip(dip_selection.index, dipole_act_quant, time_rois, axes.flat):
    ax.plot(act)
    ax.set_title(region)
    for j, cat in enumerate(stimuli['type'].unique()):
        cat_index = np.flatnonzero(stimuli['type'] == cat)
        ax.plot(cat_index[[0, -1]], act[cat_index].mean().repeat(2), color=matplotlib.cm.magma(j / 4), label=cat)

    print(region)
plt.tight_layout()

##
# Show correlation between each layer and the dipole activity
n_cols = 2
n_rows = int(np.ceil(len(dip_selection) / 2))
fig3, axes = plt.subplots(n_rows, n_cols, sharex=True, figsize=(8, 5))
for region, r, ax in zip(dip_selection.index, r1, axes.flat):
    ax.bar(layer_names[1::2], r)
    ax.set_title(region)
plt.tight_layout()


##
# Assemble all plots in an HTML report
with mne.open_report(fname.report(subject=subject)) as report:
    report.add_figs_to_section(fig1, 'Dipole timecourses', section='dipoles', replace=True)
    report.add_figs_to_section(fig2, 'Dipole item activations', section='dipoles', replace=True)
    report.add_figs_to_section(fig3, 'Correlation with model layers', section='model', replace=True)
    report.save(fname.report_html(subject=subject), overwrite=True, open_browser=False)
