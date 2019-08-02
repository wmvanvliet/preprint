import mne
import numpy as np
import pandas as pd
import unicodedata
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.io import loadmat
import rsa

def strip_accents(s):
    return ''.join(c.strip() for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

# Read in the word2vec vectors
m = loadmat('/m/nbe/scratch/redness1/semantic_features/Ginter/Ginter-300-5+5.mat')
word2vec_words = [w[0][0] for w in m['sorted']['wordsNoscand'][0, 0]]
#order = stimuli.index.get_indexer_for(word2vec_words)
word2vec = m['sorted']['mat'][0, 0]

# Read in various properties of the stimuli
stimuli = pd.read_excel(
    '/m/nbe/archive/redness1/stimuli/Red1StimuliPsychoMetrics.xlsx',
    sheet_name='Sheet1',
    skiprows=8,
    index_col=1,
)
stimuli.index.name = 'word'
stimuli['letters'] = stimuli['ITEM'].map(len)
stimuli = stimuli.loc[word2vec_words]

# Read noise cov
cov_fname = '/m/nbe/scratch/redness1/MEG/source_localized/ico4/s{subject:02d}/solutions/s{subject:02d}_alldat05-20Hz-cov.fif'
noise_cov = mne.read_cov(cov_fname.format(subject=1))

# Read in MEG data
evoked_fname = '/m/nbe/scratch/redness1/MEG/filtered0_5-20Hz_evoked/s{subject:02d}/s{subject:02d}_{word}-ave.fif'
evokeds = []
y = []
pbar = tqdm(total=20 * 123)
for i, word in enumerate(word2vec_words):
    #evoked = None
    for subject in range(20):
        ev = mne.read_evokeds(evoked_fname.format(subject=subject + 1, word=word))[0]
        ev.decimate(4)
        ev.info.normalize_proj()
        ev = mne.whiten_evoked(ev, noise_cov, diag=True)
        ev.comment = word
        y.append(i)
        #if evoked is None:
        #    evoked = ev
        #else:
        #    evoked.data += ev.data
        evokeds.append(ev)
        pbar.update(1)
    #evoked.data /= 20
    #evokeds.append(evoked)

mne.write_evokeds('/l/vanvlm1/redness1/all-ave.fif', evokeds)
np.save('/l/vanvlm1/redness1/word2vec.npy', word2vec)
stimuli.to_csv('/l/vanvlm1/redness1/stimuli.csv')

# Create image files
dpi = 96.
f = plt.figure(figsize=(128 / dpi, 128 / dpi), dpi=dpi)
for word in tqdm(stimuli['ITEM']):
    plt.clf()
    ax = f.add_axes([0, 0, 1, 1])
    fontsize = 19
    family = 'arial'
    ax.text(0.5, 0.5, word, ha='center', va='center',
            fontsize=fontsize, family=family)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.savefig('/l/vanvlm1/redness1/images/%s.JPEG' % strip_accents(word))

# rsa_word2vec = rsa.rsa_evokeds(evokeds, word2vec, spatial_radius=0.001, n_jobs=4, verbose=True)
# rsa_length = rsa.rsa_evokeds(evokeds, stimuli['letters'].values[:, None], model_dsm_metric='euclidean', spatial_radius=0.001, n_jobs=4, verbose=True)
# rsa_category = rsa.rsa_evokeds(evokeds, pd.get_dummies(stimuli['Category']).values, spatial_radius=0.001, n_jobs=4, verbose=True)
