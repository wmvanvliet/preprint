"""
Perform bandpass filtering.
"""
import argparse
import mne

from config import fname, fmin, fmax, bads, n_jobs

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='#', type=int, help='The subject to process')
parser.add_argument('run', metavar='#', type=int, help='The run to process')
args = parser.parse_args()
subject = args.subject
run = args.run
print('Processing subject:', subject)

raw = mne.io.read_raw_fif(fname.raw(subject=subject, run=run), preload=True)
raw.info['bads'] = bads[subject]
psd_raw = raw.plot_psd(tmax=60, fmax=100, show=False)

# Extract ECG and EOG events from the raw data, before any filtering. We will need these
# later during the ICA step.
eog_events = mne.preprocessing.find_eog_events(raw)
ecg_events, _, _ = mne.preprocessing.find_ecg_events(raw)
mne.write_events(fname.eog_events(subject=subject, run=run), eog_events)
mne.write_events(fname.ecg_events(subject=subject, run=run), ecg_events)

raw = mne.preprocessing.maxwell_filter(raw, st_duration=60, destination=(0, 0, 0.04))

raw = raw.filter(fmin, fmax, n_jobs=n_jobs)
psd_filtered = raw.plot_psd(tmax=60, fmax=100, show=False)

raw.save(fname.raw_filt(subject=subject, run=run), overwrite=True)

# Also create a detrended version for use with ICA later on
raw = raw.filter(1, None, n_jobs=n_jobs)
raw.save(fname.raw_detrend(subject=subject, run=run), overwrite=True)

# Add a plot of the data to the HTML report
with mne.open_report(fname.report(subject=subject)) as report:
    report.add_figs_to_section(psd_raw, f'Run {run}: PSD before bandpass filter', section='filtering',
                               replace=True)
    report.add_figs_to_section(psd_filtered, f'Run {run}: PSD after bandpass filter', section='filtering',
                               replace=True)
    report.save(fname.report_html(subject=subject), overwrite=True,
                open_browser=False)
