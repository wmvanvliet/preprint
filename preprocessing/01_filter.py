"""
Perform bandpass filtering.
"""
import argparse
import mne

from config import fname, fmin, fmax, n_runs, bads, n_jobs

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='#', type=int, help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Load the data, filter it and save the result
raw = mne.concatenate_raws([mne.io.read_raw_fif(fname.raw(subject=subject, run=run + 1))
                            for run in range(n_runs[subject])]).load_data()
raw.info['bads'] = bads[subject]
psd_raw = raw.plot_psd(tmax=60, fmax=100, show=False)

raw = raw.filter(fmin, fmax, n_jobs=n_jobs)
psd_filtered = raw.plot_psd(tmax=60, fmax=100, show=False)

raw.save(fname.raw_filt(subject=subject), overwrite=True)

# Add a plot of the data to the HTML report
with mne.open_report(fname.report(subject=subject)) as report:
    report.add_figs_to_section(psd_raw, 'PSD before bandpass filter', section='filtering',
                               replace=True)
    report.add_figs_to_section(psd_filtered, 'PSD after bandpass filter', section='filtering',
                               replace=True)
    report.save(fname.report_html(subject=subject), overwrite=True,
                open_browser=False)
