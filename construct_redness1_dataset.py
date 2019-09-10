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
stimuli = stimuli.reset_index()

# Read in MEG data
evoked_fname = '/m/nbe/scratch/redness1/MEG/filtered0_5-20Hz_evoked/s{subject:02d}/s{subject:02d}_{word}-ave.fif'
cov_fname = '/m/nbe/scratch/redness1/MEG/source_localized/ico4/s{subject:02d}/solutions/s{subject:02d}_alldat05-20Hz-cov.fif'
epochs_data = []
epochs_metadata = []
y = []
pbar = tqdm(total=20 * len(stimuli))
for subject in range(20):
    for i, word in enumerate(word2vec_words):
        ev = mne.read_evokeds(evoked_fname.format(subject=subject + 1, word=word))[0]
        noise_cov = mne.read_cov(cov_fname.format(subject=subject + 1))
        ev.decimate(5)
        ev.info.normalize_proj()
        ev = mne.whiten_evoked(ev, noise_cov, diag=True)
        epochs_data.append(ev.data)
        metadata = stimuli.iloc[i]
        metadata['label'] = i
        epochs_metadata.append(metadata)
        pbar.update(1)

epochs_metadata = pd.DataFrame(epochs_metadata)
events = np.zeros((len(epochs_data), 3), dtype=np.int)
events[:, 0] = np.arange(len(events))
events[:, 2] = epochs_metadata.label
epochs = mne.EpochsArray(np.array(epochs_data), ev.info, events, event_id={word: i for i, word in enumerate(word2vec_words)}, tmin=ev.times[0], metadata=epochs_metadata)
epochs.save('/l/vanvlm1/redness1/all-epo.fif', overwrite=True)
np.save('/l/vanvlm1/redness1/word2vec.npy', word2vec)
stimuli.to_csv('/l/vanvlm1/redness1/stimuli.csv')

# Create image files
plt.close('all')
dpi = 96.
f = plt.figure(figsize=(64 / dpi, 64 / dpi), dpi=dpi)
for word in tqdm(stimuli['word']):
    plt.clf()
    ax = f.add_axes([0, 0, 1, 1])
    fontsize = 8
    family = 'courier new'
    ax.text(0.5, 0.5, word, ha='center', va='center',
            fontsize=fontsize, family=family)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.savefig('/l/vanvlm1/redness1/images/%s.JPEG' % word)
plt.close()

# rsa_word2vec = rsa.rsa_evokeds(evokeds, word2vec, spatial_radius=0.001, n_jobs=4, verbose=True)
# rsa_length = rsa.rsa_evokeds(evokeds, stimuli['letters'].values[:, None], model_dsm_metric='euclidean', spatial_radius=0.001, n_jobs=4, verbose=True)
# rsa_category = rsa.rsa_evokeds(evokeds, pd.get_dummies(stimuli['Category']).values, spatial_radius=0.001, n_jobs=4, verbose=True)
