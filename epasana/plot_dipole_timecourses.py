"""
Compare grand-average activity at the Epasana dipoles with the model activity.
"""
import mne
import numpy as np
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt

from config import fname, subjects

metadata = []
dip_timecourses = []
dip_selection = []
for subject in tqdm(subjects):
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

groups = ['LeftOcci1', 'LeftOcciTemp2', 'LeftTemp3']
intervals = [(0.064, 0.115), (0.114, 0.200), (0.300, 0.500)]
fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(6, 3))
for group, interval, ax in zip(groups, intervals, axes.flat):
    group_meta = []
    group_tcs = []
    sel = dip_selection.query(f'group=="{group}"')
    for sub, dip, neg in zip(sel.subject, sel.dipole, sel.neg):
        tc = dip_timecourses[sub - 1][dip] * (-1 if neg else 1) * 1E9
        group_tcs.append(tc)
        df = metadata.query(f'subject=={sub}').sort_values('tif_file')
        group_meta.append(df)
    group_meta = pd.concat(group_meta, ignore_index=True)
    group_tcs = np.vstack(group_tcs)

    stimulus_types = ['noisy word', 'consonants', 'pseudoword', 'word', 'symbols']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    for stimulus_type, color in zip(stimulus_types, colors):
        tc = group_tcs[group_meta.query(f'type == "{stimulus_type}"').index].mean(axis=0)
        ax.plot(epochs.times, tc, color=color, label=stimulus_type)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlim(-0.1, 0.6)
        #ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', 0))
        ax.set_title(group)
        ax.set_xlabel('Time (s)')
plt.tight_layout()
