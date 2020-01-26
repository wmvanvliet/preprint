"""
Perform ICA rejection of EOG and ECG artefacts
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
raw.info['bads'] = bads

raw = raw.filter(1, None, n_jobs=n_jobs)

ica = mne.preprocessing.ICA(0.99).fit(raw, decim=5)
_, eog_scores = ica.find_bads_eog(raw)
eog_bads = [i for i, s in enumerate(eog_scores[0]) if abs(s) > 0.15]
eog_bads += [i for i, s in enumerate(eog_scores[1]) if abs(s) > 0.15]

_, ecg_scores = ica.find_bads_ecg(raw)
ecg_bads = [i for i, s in enumerate(ecg_scores) if abs(s) > 0.15]

ica.exclude = list(set(eog_bads + ecg_bads))
ica.save(fname.ica(subject=subject))

# Add a plot of the data to the HTML report
with mne.open_report(fname.report(subject=subject)) as report:
    report.add_figs_to_section(ica.plot_scores(eog_scores, eog_bads), 'ICA component correlation with EOG', section='filtering', replace=True)
    report.add_figs_to_section(ica.plot_scores(ecg_scores, ecg_bads), 'ICA component correlation with ECG', section='filtering', replace=True)
    report.add_figs_to_section(ica.plot_components(ica.exclude), 'Rejected ICA components', section='filtering', replace=True)
    report.save(fname.report_html(subject=subject), overwrite=True,
                open_browser=False)
