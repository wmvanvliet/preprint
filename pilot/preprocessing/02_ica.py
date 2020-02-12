"""
Perform ICA rejection of EOG and ECG artefacts
"""
import argparse
import mne

from config import fname

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='#', type=int, help='The subject to process')
parser.add_argument('run', metavar='#', type=int, help='The run to process')
args = parser.parse_args()
subject = args.subject
run = args.run
print('Processing subject:', subject)

raw = mne.io.read_raw_fif(fname.raw_detrend(subject=subject, run=run), preload=True)
ica = mne.preprocessing.ICA(0.999).fit(raw, decim=5)

eog_events = mne.read_events(fname.eog_events(subject=subject, run=run))
eog_epochs = mne.Epochs(raw, eog_events, 998, tmin=-0.5, tmax=0.5, preload=True)
eog_bads, eog_scores = ica.find_bads_eog(eog_epochs, threshold=5)

ecg_events = mne.read_events(fname.ecg_events(subject=subject, run=run))
ecg_epochs = mne.Epochs(raw, ecg_events, 999, tmin=-0.5, tmax=0.5, preload=True)
_, ecg_scores = ica.find_bads_ecg(ecg_epochs, threshold=3)
ecg_bads = [i for i, s in enumerate(ecg_scores) if abs(s) > 0.2]

ica.exclude = list(set(eog_bads + ecg_bads))
ica.save(fname.ica(subject=subject, run=run))

# Add a plot of the data to the HTML report
with mne.open_report(fname.report(subject=subject)) as report:
    report.add_figs_to_section(ica.plot_scores(eog_scores, eog_bads, show=False), f'Run {run}: ICA component correlation with EOG', section='filtering', replace=True)
    report.add_figs_to_section(ica.plot_overlay(eog_epochs.average(), eog_bads, show=False), f'Run {run}: Effect of removing EOG components', section='filtering', replace=True)
    report.add_figs_to_section(ica.plot_scores(ecg_scores, ecg_bads, show=False), f'Run {run}: ICA component correlation with ECG', section='filtering', replace=True)
    report.add_figs_to_section(ica.plot_overlay(ecg_epochs.average(), ecg_bads, show=False), f'Run {run}: Effect of removing ECG components', section='filtering', replace=True)
    report.add_figs_to_section(ica.plot_components(ica.exclude, show=False), f'Run {run}: Rejected ICA components', section='filtering', replace=True)
    report.save(fname.report_html(subject=subject), overwrite=True, open_browser=False)
