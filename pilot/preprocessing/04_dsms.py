"""
Compute sensor-level DSMs in a searchlight pattern.
"""
import argparse
import mne_rsa
import mne
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from config import fname, n_jobs

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='#', type=int, help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

epochs = mne.read_epochs(fname.epochs(subject=subject))
epochs = epochs[['word', 'symbols', 'consonants']]
epochs.crop(0, 0.8).resample(100).pick_types(meg='grad')
stimuli = epochs.metadata.groupby('text').agg('first').sort_values('event_id')
stimuli['y'] = np.arange(len(stimuli))
metadata = pd.merge(epochs.metadata, stimuli[['y']], left_on='text', right_index=True).sort_index()
assert np.array_equal(metadata.event_id.values.astype(int), epochs.events[:, 2])

def compute_dsms(picks):
    return list(mne_rsa.dsm_epochs(
        epochs,
        y=metadata['y'],
        spatial_radius=0.04,
        temporal_radius=0.05,
        dist_metric='correlation',
        picks=picks,
    ))

dsms = Parallel(n_jobs=n_jobs, verbose=True)(
    delayed(compute_dsms)(p) for p in range(epochs.info['nchan'])
)

np.savez_compressed(
    fname.dsms(subject=subject),
    dsms=dsms,
    ch_names=epochs.ch_names,
    times=epochs.times[5:-4],
)
