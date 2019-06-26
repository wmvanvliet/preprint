import mne
from scipy.io import loadmat, savemat
import numpy as np
import pandas as pd
import unicodedata
from tqdm import tqdm
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
    '/m/nbe/scratch/redness1/stimuli/StimulusSelection20140610.xlsx',
    sheet_name='Redness',
    usecols=[2,3,4,8,9],
    index_col=0
)
stimuli.index = stimuli['Finnish'].map(strip_accents)
stimuli.index.name = 'word'
stimuli['letters'] = stimuli['Finnish'].map(len)
stimuli = stimuli.loc[word2vec_words]

# Read noise cov
cov_fname = '/m/nbe/scratch/redness1/MEG/source_localized/ico4/s{subject:02d}/solutions/s{subject:02d}_alldat05-20Hz-cov.fif'
noise_cov = mne.read_cov(cov_fname.format(subject=1))

# Read in MEG data
evoked_fname = '/m/nbe/scratch/redness1/MEG/filtered0_5-20Hz_evoked/s{subject:02d}/s{subject:02d}_{word}-ave.fif'
evokeds = []
pbar = tqdm(total=20 * 123)
for word in word2vec_words:
    evoked = None
    for subject in range(20):
        ev = mne.read_evokeds(evoked_fname.format(subject=subject + 1, word=word))[0]
        ev.decimate(4)
        #ev.info.normalize_proj()
        #ev = mne.whiten_evoked(ev, noise_cov)
        if evoked is None:
            evoked = ev
        else:
            evoked.data += ev.data
        pbar.update(1)
    evoked.data /= 20
    evokeds.append(evoked)

#rsa_word2vec = rsa.rsa_evokeds(evokeds, word2vec, spatial_radius=0.001, n_jobs=4, verbose=True)
#rsa_length = rsa.rsa_evokeds(evokeds, stimuli['letters'].values[:, None], model_dsm_metric='euclidean', spatial_radius=0.001, n_jobs=4, verbose=True)
#rsa_category = rsa.rsa_evokeds(evokeds, pd.get_dummies(stimuli['Category']).values, spatial_radius=0.001, n_jobs=4, verbose=True)
