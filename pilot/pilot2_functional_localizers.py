import mne
import posthoc
import pickle
import rsa
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from matplotlib import pyplot as plt

epochs = mne.read_epochs('../data/pilot_data/pilot2/pilot2_epo.fif')
epochs = epochs[['word', 'symbols', 'consonants']]
epochs.crop(0, 0.8).resample(100).pick_types(meg='grad')
stimuli = epochs.metadata.groupby('text').agg('first').sort_values('event_id')
stimuli['y'] = np.arange(len(stimuli))
metadata = pd.merge(epochs.metadata, stimuli[['y']], left_on='text', right_index=True).sort_index()
assert np.array_equal(metadata.event_id.values.astype(int), epochs.events[:, 2])

vis_template = mne.combine_evoked([epochs['noise_level==0.5'].average(), epochs['noise_level==0.2'].average()], weights=[1, -1])
letter_template = mne.combine_evoked([epochs['word'].average(), epochs['consonants'].average(), epochs['symbols'].average()], weights=[1, 1, -2])
word_template = mne.combine_evoked([epochs['word'].average(), epochs['consonants'].average(), epochs['symbols'].average()], weights=[2, -1, -1])

times=[0.05, 0.1, 0.15, 0.17, 0.2, 0.25, 0.3, 0.4, 0.5]
vis_template.plot_topomap(times, title='Visual template')
letter_template.plot_topomap(times, title='Letter template')
word_template.plot_topomap(times, title='Word template')

def make_X(epochs):
    """Construct an n_samples x n_channels matrix from an mne.Epochs object."""
    #X = epochs.get_data().transpose(0, 2, 1).reshape(-1, epochs.info['nchan'])
    X = epochs.get_data().reshape(len(epochs), -1)
    return X
X = make_X(epochs)

def pattern_modifier(pattern, X_train=None, y_train=None, mu=0.36, sigma=0.06):
    pattern = pattern.reshape(204, 80)
    
    # Define mu and sigma in samples
    mu = np.searchsorted(epochs.times, mu)
    sigma = sigma  * epochs.info['sfreq']
    
    # Formula for Gaussian curve
    kernel = np.exp(-0.5 * ((np.arange(80) - mu) / sigma) ** 2)
    
    return (pattern * kernel).ravel()

def plot_mod_pattern(ev, mu, sigma):
    pat = pattern_modifier(ev.data, mu=mu, sigma=sigma)
    ev = mne.EvokedArray(pat.reshape(ev.data.shape), ev.info)
    ev.plot()

m = posthoc.Beamformer(pattern_modifier(vis_template.data, mu=0.1, sigma=0.05),
                       cov=posthoc.cov_estimators.ShrinkageKernel(0.95)).fit(X)
y_vis = m.transform(make_X(epochs))
plt.figure()
plt.bar([1, 2, 3, 4, 5, 6], [
    m.transform(make_X(epochs['noise_level==0.2'])).mean(),
    m.transform(make_X(epochs['noise_level==0.35'])).mean(),
    m.transform(make_X(epochs['noise_level==0.5'])).mean(),
    m.transform(make_X(epochs['word'])).mean(),
    m.transform(make_X(epochs['consonants'])).mean(),
    m.transform(make_X(epochs['symbols'])).mean(),
])
plt.axhline(0, color='black')
plt.title('Visual template')
plt.xticks([1, 2, 3, 4, 5, 6], ['noise=0.2', 'noise=0.35', 'noise=0.5', 'word', 'consonants',' symbols'])

m = posthoc.Beamformer(pattern_modifier(letter_template.data, mu=0.17, sigma=0.07),
                       cov=posthoc.cov_estimators.ShrinkageKernel(0.95)).fit(X)
y_letter = m.transform(make_X(epochs))
plt.figure()
plt.bar([1, 2, 3, 4, 5, 6], [
    m.transform(make_X(epochs['noise_level==0.2'])).mean(),
    m.transform(make_X(epochs['noise_level==0.35'])).mean(),
    m.transform(make_X(epochs['noise_level==0.5'])).mean(),
    m.transform(make_X(epochs['word'])).mean(),
    m.transform(make_X(epochs['consonants'])).mean(),
    m.transform(make_X(epochs['symbols'])).mean(),
])
plt.axhline(0, color='black')
plt.title('Letter template')
plt.xticks([1, 2, 3, 4, 5, 6], ['noise=0.2', 'noise=0.35', 'noise=0.5', 'word', 'consonants',' symbols'])

m = posthoc.Beamformer(pattern_modifier(word_template.data, mu=0.4, sigma=0.1),
                       cov=posthoc.cov_estimators.ShrinkageKernel(0.95)).fit(X)
y_word = m.transform(make_X(epochs))
plt.figure()
plt.bar([1, 2, 3, 4, 5, 6], [
    m.transform(make_X(epochs['noise_level==0.2'])).mean(),
    m.transform(make_X(epochs['noise_level==0.35'])).mean(),
    m.transform(make_X(epochs['noise_level==0.5'])).mean(),
    m.transform(make_X(epochs['word'])).mean(),
    m.transform(make_X(epochs['consonants'])).mean(),
    m.transform(make_X(epochs['symbols'])).mean(),
])
plt.axhline(0, color='black')
plt.title('Word template')
plt.xticks([1, 2, 3, 4, 5, 6], ['noise=0.2', 'noise=0.35', 'noise=0.5', 'word', 'consonants',' symbols'])

# y_word = m.transform(make_X(epochs['word']))
# y_consonants = m.transform(make_X(epochs['consonants']))
# y_symbols = m.transform(make_X(epochs['symbols']))

#model_name = 'vgg_first_imagenet64_then_tiny-words-noisy_tiny-imagenet'
model_name = 'vgg_first_imagenet64_then_tiny-words_w2v'
with open(f'../data/dsms/pilot2_{model_name}_dsms.pkl', 'rb') as f:
    dsm_models = pickle.load(f)
    dsms = dsm_models['dsms']
    dsm_names = dsm_models['dsm_names']

rsa_vis = rsa.rsa_array(y_vis, dsms, y=metadata['y'], data_dsm_metric='euclidean', rsa_metric='kendall-tau-a')[0, 0, 0]
rsa_letter = rsa.rsa_array(y_letter, dsms, y=metadata['y'], data_dsm_metric='euclidean', rsa_metric='kendall-tau-a')[0, 0, 0]
rsa_word = rsa.rsa_array(y_word, dsms, y=metadata['y'], data_dsm_metric='euclidean', rsa_metric='kendall-tau-a')[0, 0, 0]

f = plt.figure(figsize=(10, 4))
ax = f.subplots(1, 3, sharey=True, sharex=True)
ax[0].bar(range(len(rsa_vis)), abs(rsa_vis))
ax[0].set_xticks(range(len(rsa_vis)))
ax[0].set_xticklabels(dsm_names, rotation=90)
ax[0].set_title('Visual template')

ax[1].bar(range(len(rsa_letter)), abs(rsa_letter))
ax[1].set_xticks(range(len(rsa_letter)))
ax[1].set_xticklabels(dsm_names, rotation=90)
ax[1].set_title('Letter template')

ax[2].bar(range(len(rsa_word)), abs(rsa_word))
ax[2].set_xticks(range(len(rsa_word)))
ax[2].set_xticklabels(dsm_names, rotation=90)
ax[2].set_title('Word template')

plt.ylim(0, 0.43)
plt.tight_layout()


import numpy as np

from sklearn.linear_model import LinearRegression

def scorer(model, X, y):
    return pearsonr(model.predict(X), y)[0]

model = posthoc.WorkbenchOptimizer(
    LinearRegression(),
    cov=posthoc.cov_estimators.ShrinkageKernel(0.95),
    pattern_modifier=pattern_modifier,
    pattern_param_x0=[0.1, 0.05],
    pattern_param_bounds=[(0, 0.8), (0.005, 0.5)],
    scoring=scorer,
)
model.fit(X, epochs.metadata['noise_level'].values)
