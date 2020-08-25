import sys
import mne
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import zscore, rankdata
from tqdm import tqdm
from joblib import Parallel, delayed

from config import fname

subject = int(sys.argv[1])

stimuli = pd.read_csv(fname.stimulus_selection)
stimuli['type'] = stimuli['type'].astype('category')
info = mne.io.read_info(fname.info_102)

epochs = mne.read_epochs(fname.epochs(subject=subject))

# Drop any stimuli for which there are no corresponding epochs
stimuli = stimuli[stimuli.tif_file.isin(epochs.metadata.tif_file)]

# Re-order epochs to be in the order of the stimuli
epochs.metadata['epoch_index'] = np.arange(len(epochs))
epochs = epochs[epochs.metadata.sort_values('tif_file').epoch_index.values]

# Load model layer activations
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

# Estimate source activation for the epochs.
snr = 3  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2
inv = mne.minimum_norm.read_inverse_operator(fname.inv(subject=subject))
inv = mne.minimum_norm.prepare_inverse_operator(inv, nave=len(epochs), lambda2=lambda2)
morph = mne.read_source_morph(fname.morph(subject=subject))
stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2=lambda2, prepared=True, return_generator=True, verbose=False, pick_ori='normal')

# 32-bit instead of 64-bit to save memory
stc_data = np.zeros((len(epochs), 20484, 60), dtype=np.float32)
for i, stc in tqdm(enumerate(stcs), total=len(epochs)):
    stc = morph.apply(stc)
    # Crop and resample to save memory.
    stc = stc.crop(0, 0.6).bin(0.01)
    times = stc.times.copy()
    vertices = stc.vertices
    stc_data[i] = stc.data.astype(np.float32)

# Compute correlations in parallel
def compute_r(layer_activity, stc_data):
    return spearmanr(layer_activity, stc_data, x_axis=1, y_axis=0)
r = Parallel(n_jobs=4)(delayed(compute_r)(layer_activity, stc_data[:, :, i])
                       for i in tqdm(range(len(times))))
r = np.array(r).transpose(1, 2, 0)

np.savez_compressed(fname.stc_layer_corr(subject=subject), r=r, times=times, vertices_lh=vertices[0], vertices_rh=vertices[1])
