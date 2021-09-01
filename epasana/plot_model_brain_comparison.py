"""
Create the plots in which the behavior of the layers in the model are compared
to the behavior of the epasana dipoles.
"""
import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.stats import zscore

from config import fname

# Brain landmarks to show
selected_landmarks = ['LeftOcci1', 'LeftOcciTemp2', 'LeftTemp3']

# Stimulus types in the order in which they should be plotted
stimulus_types = ['word', 'pseudoword', 'consonants', 'symbols', 'noisy word']

# The subjects to plot the data for
subjects = range(1, 16)

# Read in the epochs metadata and dipole information from each subject
metadata = []
dip_timecourses = []
dip_selection = []
for subject in tqdm(subjects):
    # Metadata: what stimulus was shown during each epoch?
    epochs = mne.read_epochs(fname.epochs(subject=subject), preload=False)
    m = epochs.metadata.copy()
    m['subject'] = subject
    m['epoch_index'] = np.arange(len(epochs))
    metadata.append(m)

    # Dipole timecourses for each epoch (make sure it's in the correct order!)
    dip_t = np.load(fname.dip_timecourses(subject=subject))['proj']
    dip_t = dip_t[:, np.argsort(epochs.metadata.tif_file), :]
    dip_timecourses.append(dip_t)

    # Select the dipoles corresponding to the landmarks.
    # These are different for each subject.
    dip_sel = pd.read_csv(fname.dip_selection(subject=subject), sep='\t')
    dip_sel['subject'] = subject
    dip_selection.append(dip_sel)
metadata = pd.concat(metadata, ignore_index=True)
dip_selection = pd.concat(dip_selection, ignore_index=True)

# For each landmark, we analyze a different time interval
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

## For each subject, compute mean dipole activation for each landmark
landmark_acts = []
for subject, sub_dip_timecourses in zip(subjects, dip_timecourses):
    dip_info = dip_selection.query(f'subject=="{subject}"')
    landmark_act = metadata.query(f'subject=={subject}').sort_values('tif_file')
    for dip, landmark, neg in zip(dip_info.dipole, dip_info.group, dip_info.neg):
        tc = sub_dip_timecourses[dip] * (-1 if neg else 1)
        quant = zscore(tc[:, time_rois[landmark]].mean(axis=1))
        landmark_act[landmark] = quant
    landmark_acts.append(landmark_act)
landmark_acts = pd.concat(landmark_acts, ignore_index=True)
landmark_acts.to_csv(fname.dip_activation, index=False)

## For three landmarks, show the mean dipole activation for each stimulus.
mean_acts = []
fig, axes = plt.subplots(1, len(selected_landmarks), sharex=True, sharey=True, figsize=(4.4, 3))
for landmark, ax in zip(selected_landmarks, axes.flat):
    x_offset = 0
    for i, stimulus_type in enumerate(stimulus_types):
        mean_act = landmark_acts.query(f'type=="{stimulus_type}"').groupby('tif_file').agg('mean')[landmark]
        mean_acts.append(mean_act)
        ax.scatter(np.arange(len(mean_act)) + x_offset, mean_act, s=1, alpha=0.2)
        ax.plot([x_offset, x_offset + len(mean_act) - 1], mean_act.mean().repeat(2), color=plt.get_cmap('tab10').colors[i], label=stimulus_type)
        ax.xaxis.set_visible(False)
        ax.set_facecolor('#eee')
        ax.set_facecolor('#fafbfc')
        for pos in ['top', 'bottom', 'left', 'right']:
            ax.spines[pos].set_visible(False)
        x_offset += len(mean_act)
    if landmark == selected_landmarks[0]:
        ax.legend()
        ax.set_ylabel('Activation (z-scores)')
        #ax.set_ylabel('Activation (fT)')
    else:
        ax.yaxis.set_visible(False)
    if landmark == 'LeftOcci1':
        ax.set_title('Left Occi.\n65-115 ms', fontsize=10)
        ax.spines['left'].set_visible(True)
    elif landmark == 'LeftOcciTemp2':
        ax.set_title('Left Occi.Temp.\n140-200 ms', fontsize=10)
    elif landmark == 'LeftTemp3':
        ax.set_title('Left Temp.\n300-500 ms', fontsize=10)
plt.tight_layout()
mean_acts = np.array(mean_acts)

## Load model layer activations
import pickle
from scipy.stats import rankdata
model_name = 'vgg11_first_imagenet_then_epasana-10kwords_noise'
#model_name = 'vgg11_first_imagenet_then_epasana-10kwords_epasana-nontext_imagenet256'
#model_name = 'vgg11_epasana-10kwords_noise'
with open(fname.model_dsms(name=model_name), 'rb') as f:
    d = pickle.load(f)
layer_activity = zscore(np.array(d['layer_activity'])[1::2], axis=1)
layer_names = d['dsm_names'][:-4][1::2]
layer_acts = pd.DataFrame(layer_activity.T, columns=layer_names)

stimuli = pd.read_csv(fname.stimulus_selection)
stimuli['type'] = stimuli['type'].astype('category')

## Show the behavior of each model layer for each stimulus
fig, axes = plt.subplots(1, len(layer_activity), sharex=True, sharey=True, figsize=(10, 3))
for i, (act, ax) in enumerate(zip(layer_activity, axes.flat)):
    x_offset = 0
    for j, cat in enumerate(stimulus_types):
        cat_index = np.flatnonzero(stimuli['type'] == cat)
        selection = act[cat_index]
        ax.scatter(np.arange(len(selection)) + x_offset, selection, s=1, alpha=0.2)
        ax.plot([x_offset, x_offset+len(selection) - 1], selection.mean().repeat(2), color=plt.get_cmap('tab10').colors[j], label=cat)
        ax.xaxis.set_visible(False)
        ax.set_facecolor('#eee')
        ax.set_facecolor('#fafbfc')
        ax.set_title(f'{layer_names[i]}\n', fontsize=10)
        for pos in ['top', 'bottom', 'left', 'right']:
            ax.spines[pos].set_visible(False)
        if i == 0:
            ax.legend()
            ax.set_ylabel('Activation (z-scores)')
            ax.spines['left'].set_visible(True)
        else:
            ax.yaxis.set_visible(False)
        x_offset += len(selection)
plt.tight_layout()

##
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

r1 = pearsonr(mean_acts, layer_activity)
r2 = spearmanr(mean_acts, layer_activity)
fig, axes = plt.subplots(1, 3, sharex=True)
for r, ax, landmark in zip(r1, axes.flat, selected_landmarks):
    ax.bar(np.arange(len(r)), r)
    ax.set_title(landmark)


## Save giant CSV file
data = landmark_acts.join(stimuli[['tif_file']].join(layer_acts).set_index('tif_file'), on='tif_file')
data.to_csv(fname.brain_model_comparison)

#     (Intercept)   fc2_relu
#  2  -0.0008408194 0.20664959
#  3  -0.0008408194 0.21845941
#  4  -0.0008408194 0.17940047
#  6  -0.0008408194 0.03159088
#  7  -0.0008408194 0.17615774
#  8  -0.0008408194 0.09583091
#  9  -0.0008408194 0.13515399
#  10 -0.0008408194 0.24666465
#  11 -0.0008408194 0.19468480
#  13 -0.0008408194 0.06752757
#  14 -0.0008408194 0.16153019
#  15 -0.0008408194 0.13566348
