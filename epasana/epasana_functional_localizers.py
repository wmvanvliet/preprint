import mne
import posthoc
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

epochs = mne.read_epochs('../data/epasana/items-epo.fif')
stimuli = pd.read_csv('stimulus_selection.csv')
stimuli['type'] = stimuli['type'].astype('category')
info = mne.io.read_info('info_102.fif')

ga = []
for t in stimuli.type.unique():
    ev = epochs[f'type=="{t}"'].average()
    grads_comb = np.linalg.norm(ev.data.reshape(2, 102, 80), axis=0)
    ev = mne.EvokedArray(grads_comb, info, tmin=ev.times[0])
    ev.comment = t
    ga.append(ev)


model_name = 'vgg11_first_imagenet_then_epasana-10kwords_epasana-nontext'
with open(f'../data/dsms/epasana_{model_name}_dsms.pkl', 'rb') as f:
    d = pickle.load(f)
layer_activity = d['layer_activity']
layer_names = d['dsm_names'][:-4]

noise_sens = stimuli['type'].isin(['noisy word']).values.astype(int)[:, None]
letter_sens = stimuli['type'].isin(['consonants', 'word', 'pseudoword']).values.astype(int)[:, None]
word_sens = stimuli['type'].isin(['word', 'pseudoword']).values.astype(int)[:, None]

r = mne.stats.linear_regression(epochs, np.hstack([noise_sens, letter_sens, word_sens]), ['noise', 'letter', 'word'])
#r = mne.stats.linear_regression(epochs, noise_sens, ['noise'])
noise_template = r['noise'].beta
noise_template.comment = 'noise template'
#r = mne.stats.linear_regression(epochs, letter_sens, ['letter'])
letter_template = r['letter'].beta
letter_template.comment = 'letter template'
#r = mne.stats.linear_regression(epochs, word_sens, ['word'])
word_template = r['word'].beta
word_template.comment = 'word template'

times=[0.05, 0.1, 0.15, 0.17, 0.2, 0.25, 0.3, 0.4, 0.5]
noise_template.plot_topomap(times, title='Noise template')
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

m = posthoc.Beamformer(noise_template.data.ravel(),
                       cov=posthoc.cov_estimators.ShrinkageKernel(0.05)).fit(X)
y_vis = m.transform(make_X(epochs))
plt.figure()
plt.bar([1, 2, 3, 4, 5], [
    m.transform(make_X(epochs['type=="noisy word"'])).mean(),
    m.transform(make_X(epochs['type=="consonants"'])).mean(),
    m.transform(make_X(epochs['type=="pseudoword"'])).mean(),
    m.transform(make_X(epochs['type=="word"'])).mean(),
    m.transform(make_X(epochs['type=="symbols"'])).mean(),
])
plt.axhline(0, color='black')
plt.title('Noise template')
plt.xticks([1, 2, 3, 4, 5], ['noisy word', 'consonants', 'pseudoword', 'word', 'symbols'])

m = posthoc.Beamformer(letter_template.data.ravel(),
                       cov=posthoc.cov_estimators.ShrinkageKernel(0.05)).fit(X)
y_vis = m.transform(make_X(epochs))
plt.figure()
plt.bar([1, 2, 3, 4, 5], [
    m.transform(make_X(epochs['type=="noisy word"'])).mean(),
    m.transform(make_X(epochs['type=="consonants"'])).mean(),
    m.transform(make_X(epochs['type=="pseudoword"'])).mean(),
    m.transform(make_X(epochs['type=="word"'])).mean(),
    m.transform(make_X(epochs['type=="symbols"'])).mean(),
])
plt.axhline(0, color='black')
plt.title('Letter template')
plt.xticks([1, 2, 3, 4, 5], ['noisy word', 'consonants', 'pseudoword', 'word', 'symbols'])

m = posthoc.Beamformer(word_template.data.ravel(),
                       cov=posthoc.cov_estimators.ShrinkageKernel(0.05)).fit(X)
y_vis = m.transform(make_X(epochs))
plt.figure()
plt.bar([1, 2, 3, 4, 5], [
    m.transform(make_X(epochs['type=="noisy word"'])).mean(),
    m.transform(make_X(epochs['type=="consonants"'])).mean(),
    m.transform(make_X(epochs['type=="pseudoword"'])).mean(),
    m.transform(make_X(epochs['type=="word"'])).mean(),
    m.transform(make_X(epochs['type=="symbols"'])).mean(),
])
plt.axhline(0, color='black')
plt.title('Word template')
plt.xticks([1, 2, 3, 4, 5], ['noisy word', 'consonants', 'pseudoword', 'word', 'symbols'])
