"""
Cut epochs and apply ICA to them. Store evokeds.
"""
import argparse
import mne
import pandas as pd
import numpy as np

from config import fname, event_id, n_runs

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='#', type=int, help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

raw = mne.io.read_raw_fif(fname.raw_filt(subject=subject), preload=True)

events = mne.find_events(raw, min_duration=0.005)

# Discard erroneous events
events = events[events[:, 2] >= 10]
events = events[~np.hstack([False, ((events[:-1, 2] == 40) & (events[1:, 2] == 40))])]

epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=1.0, baseline=(-0.2, 0), preload=True)
fig_evoked_raw = epochs.average().plot(show=False)

# Apply ICA to the epochs to filter out EOG and ECG artifacts
ica = mne.preprocessing.read_ica(fname.ica(subject=subject))
fig_ica_props = ica.plot_properties(epochs, ica.exclude, show=False)
epochs = ica.apply(epochs)
fig_evoked_clean = epochs.average().plot(show=False)

# Load metadata and save it in the Epochs object
if subject == 1:
    # Load Presentation logfile and extract the presented stimulus
    metadata = [pd.read_csv(fname.log(subject=subject, run=run + 1), sep='\t', skiprows=3) for run in range(n_runs[subject])]
    metadata = pd.concat(metadata, ignore_index=True)
    metadata = metadata.loc[metadata['Code'] != 'fixation']
    metadata = metadata.reset_index(drop=True)[['Code']]
else:
    # Load metadata from stimulus csv files
    metadata = pd.concat([pd.read_csv(fname.stimuli(subject=subject, run=run + 1), index_col=0)
                          for run in range(n_runs[subject])], ignore_index=True)
    # Ugly hack to add rows for question stimuli
    metadata2 = metadata.copy()
    metadata2.index = pd.Index(np.flatnonzero(epochs.events[:, 2] < 40))
    metadata2 = metadata2.reindex(index=pd.Index(np.arange(len(epochs))))
    metadata2.loc[np.flatnonzero(epochs.events[:, 2] == 40)] = metadata.query('question_asked')
    metadata2.loc[np.flatnonzero(epochs.events[:, 2] == 40), 'type'] = 'question'
    metadata2.loc[np.flatnonzero(epochs.events[:, 2] == 40), 'rotation'] = 0
    metadata2.loc[np.flatnonzero(epochs.events[:, 2] == 40), 'noise_level'] = 0
    metadata2.loc[np.flatnonzero(epochs.events[:, 2] == 40), 'text'] = metadata2.loc[np.flatnonzero(epochs.events[:, 2] == 40), 'question']
    metadata = metadata2
epochs.metadata = metadata
epochs.save(fname.epochs(subject=subject), overwrite=True)

# Save evokeds
evokeds = [epochs[cond].average() for cond in event_id.keys()]
mne.write_evokeds(fname.evoked(subject=subject), evokeds)

# Add a plot of the data to the HTML report
with mne.open_report(fname.report(subject=subject)) as report:
    report.add_slider_to_section(fig_ica_props , ['Component %d' % c for c in ica.exclude], title="ICA component properties", section='filtering', replace=True)
    report.add_figs_to_section(fig_evoked_raw, 'Evoked before ICA', section='filtering', replace=True)
    report.add_figs_to_section(fig_evoked_clean, 'Evoked after ICA', section='filtering', replace=True)
    report.save(fname.report_html(subject=subject), overwrite=True,
                open_browser=False)
