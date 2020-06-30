import mne
import mne_rsa
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

# Path to the epasana dataset on scratch
data_path = '/m/nbe/scratch/epasana'

# Read epochs and create the stimulus order. Drop epochs that do no have proper metadata.
stimuli = pd.read_csv('stimulus_selection.csv', index_col=0)

all_epochs = []
for subject in range(1, 16):
    print(f'Loading subject {subject}...', flush=True)
    epochs = mne.read_epochs(f'{data_path}/bids/derivatives/marijn/sub-{subject:02d}/meg/sub-{subject:02d}-epo.fif')
    epochs.crop(0, 0.8).resample(100).pick_types(meg='grad')
    epochs.metadata = epochs.metadata.join(stimuli['y'], on='tif_file')
    epochs.drop(np.flatnonzero(np.isnan(epochs.metadata.y).values))
    all_epochs.append(epochs)
epochs = mne.concatenate_epochs(all_epochs)
assert len(epochs) == (560 * 15)

# We're going to be computing lots of big DSMs and the result will not fit
# comfortably in memory. Hence we'll be streaming the results to disk using a
# memmap-ed file.
dsms = np.memmap('/tmp/dsms.npy', shape=(102, 70, 172578), mode='w+', dtype=np.float64)

def compute_dsms_ch(index, channel):
    """Compute DSMs for a single channel and store them in the memmap-ed dsms array.

    Parameters
    ----------
    index : int
        Integer index where in the memmap-ed array to write the result.
    channel : int
        Integer index of channel to use as center of the searchlight patch.
    """
    dsms[index] = list(mne_rsa.dsm_epochs(
        epochs,
        y=epochs.metadata['y'],
        spatial_radius=0.04,
        temporal_radius=0.05,
        dist_metric='correlation',
        picks=channel,
        n_folds=5,
    ))

# Compute DSMs in parallel. Each channel gets assigned its own worker using the
# helper function above.
channels = range(0, epochs.info['nchan'], 2)  # Only use one of the two gradiometers as patch center
Parallel(n_jobs=4, verbose=True)(
    delayed(compute_dsms_ch)(i, c) for i, c in enumerate(tqdm(channels))
)

# Collect the DSMs into a more convenient file format
print('Saving result to scratch folder...', flush=True)
np.savez_compressed(
    f'{data_path}/bids/derivatives/reading_models/epasana-dsms.npz',
    dsms=dsms,
    ch_names=epochs.ch_names,
    times=epochs.times[5:-4],
)
