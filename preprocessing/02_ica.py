"""
Perform ICA rejection of EOG and ECG artefacts
"""
import argparse
import mne

from config import fname, n_jobs

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='#', type=int, help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

raw = mne.io.read_raw_fif(fname.raw_detrend(subject=subject), preload=True)
ica = mne.preprocessing.ICA(0.9).fit(raw, decim=10)

eog_epochs = mne.preprocessing.create_eog_epochs(raw).decimate(5)
eog_bads, eog_scores = ica.find_bads_eog(eog_epochs, threshold=5)

ecg_epochs = mne.preprocessing.create_ecg_epochs(raw).decimate(5)
ecg_bads, ecg_scores = ica.find_bads_ecg(ecg_epochs, threshold=2)

#eog_bads = [i for i, s in enumerate(eog_scores[0]) if abs(s) > 0.2]
#eog_bads += [i for i, s in enumerate(eog_scores[1]) if abs(s) > 0.2]

#_, ecg_scores = ica.find_bads_ecg(raw)
#ecg_bads = [i for i, s in enumerate(ecg_scores) if abs(s) > 0.15]

ica.exclude = list(set(eog_bads + ecg_bads))
ica.save(fname.ica(subject=subject))

# Add a plot of the data to the HTML report
with mne.open_report(fname.report(subject=subject)) as report:
    report.add_figs_to_section(ica.plot_scores(eog_scores, eog_bads, show=False), 'ICA component correlation with EOG', section='filtering', replace=True)
    report.add_figs_to_section(ica.plot_scores(ecg_scores, ecg_bads, show=False), 'ICA component correlation with ECG', section='filtering', replace=True)
    report.add_figs_to_section(ica.plot_components(ica.exclude, show=False), 'Rejected ICA components', section='filtering', replace=True)
    report.save(fname.report_html(subject=subject), overwrite=True,
                open_browser=False)
