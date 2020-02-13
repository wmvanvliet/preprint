import mne
from matplotlib import pyplot as plt

times = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
rsa_results = mne.read_evokeds('../data/pilot_data/pilot2/pilot2_rsa_results-ave.fif')
rsa_names = [r.comment for r in rsa_results]

rsa_results[rsa_names.index('Noise level')].plot_topomap(
    times, units=dict(grad='Kendall tau-a'), scalings=dict(grad=1), vmax=0.3, title='Noise level')
plt.savefig('../figures/pilot2_rsa_noise-level.pdf')
rsa_results[rsa_names.index('Letters only')].plot_topomap(
    times, units=dict(grad='Kendall tau-a'), scalings=dict(grad=1), vmax=0.3, title='Letters only')
plt.savefig('../figures/pilot2_rsa_letters-only.pdf')
rsa_results[rsa_names.index('Words only')].plot_topomap(
    times, units=dict(grad='Kendall tau-a'), scalings=dict(grad=1), vmax=0.3, title='Words only')
plt.savefig('../figures/pilot2_rsa_words-only.pdf')

rsa_results[rsa_names.index('Feature 1')].plot_topomap(
    times, units=dict(grad='Kendall tau-a'), scalings=dict(grad=1), vmax=0.3, title='Feature 1')
plt.savefig('../figures/pilot2_rsa_feature-1.pdf')
rsa_results[rsa_names.index('Classifier 1')].plot_topomap(
    times, units=dict(grad='Kendall tau-a'), scalings=dict(grad=1), vmax=0.3, title='Classifier 1')
plt.savefig('../figures/pilot2_rsa_class-1.pdf')
