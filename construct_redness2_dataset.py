import mne
import numpy as np
import pandas as pd
import unicodedata
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.io import loadmat
import subprocess

def strip_accents(s):
    return ''.join(c.strip() for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

# Read in various properties of the stimuli
stimuli = pd.read_csv('/m/nbe/project/redness2/stimuli/Red2StimuliPsychoMetrics.csv')
stimuli = stimuli.set_index('ITEM')
stimuli.index.name = 'word'
stimuli['letters'] = stimuli.index.map(len)
stimuli.to_csv('/l/vanvlm1/redness2/stimuli.csv')

# Read in the word2vec vectors
m = loadmat('/m/nbe/scratch/redness2/semantic_features/Ginter-300-5+5.mat')
word2vec_words = [w[0][0] for w in m['sorted']['wordsNoscand'][0, 0]]
#order = stimuli.index.get_indexer_for(word2vec_words)
word2vec = m['sorted']['mat'][0, 0]
sel = [word2vec_words.index(w) for w in stimuli.index if w in word2vec_words]
word2vec = word2vec[sel]
word2vec_words = [word2vec_words[s] for s in sel]
np.save('/l/vanvlm1/redness2/word2vec.npy', word2vec)

all_epochs = []
for subject in tqdm([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20]):
    e = mne.read_epochs('/m/nbe/scratch/redness2/marijn/derivatives/sub-{subject:02d}/sub-{subject:02d}-task-naming_filt-ica_epo.fif'.format(subject=subject), preload=True)
    e.metadata = e.metadata.join(stimuli, on='word')
    e.decimate(2)
    e.pick_types(meg='mag')
    all_epochs.append(e)
all_epochs = mne.concatenate_epochs(all_epochs)

# Compute class labels
ref = all_epochs.metadata.drop_duplicates()
ref = ref.sort_values(['modality', 'word', 'filename', 'font']).reset_index(drop=True)
index = dict()
for i, row in ref.iterrows():
    index[tuple(row[:4])] = i
all_epochs.metadata['label'] = [index[tuple(row[:4])] for _, row in all_epochs.metadata.iterrows()]

# Compute noise cov
noise_cov = mne.compute_covariance(all_epochs, tmin=-0.2)
noise_cov.save('/l/vanvlm1/redness2/noise-cov.fif')

# Save epochs
all_epochs.save('/l/vanvlm1/redness1/all-epo.fif', overwrite=True)

# Create image files
for label, meta in tqdm(all_epochs.metadata.query('modality=="wri"').groupby('label').aggregate('first').iterrows(), total=300):
    fonts = {
        'Ubuntu Mono': '/u/45/vanvlm1/unix/.fonts/UbuntuMono-R.ttf',
        'Courier': 'Courier',
        'Luxi Mono Regular': '/u/45/vanvlm1/unix/.fonts/luximr.ttf',
        'Lucida Console': '/u/45/vanvlm1/unix/.fonts/LucidaConsole-R.TTF',
        'Lekton': '/u/45/vanvlm1/unix/.fonts/Lekton-Regular.ttf',
    }

    subprocess.call([
        'convert',
        '-background', 'white',
        '-fill', 'black',
        '-font', fonts[meta.font],
        '-pointsize', '19',
        '-size', '128x128',
        '-gravity', 'center',
        'label:%s' % meta.word,
        '/l/vanvlm1/redness2/images/%d.JPEG' % label
    ])

for label, meta in tqdm(all_epochs.metadata.query('modality=="pic"').groupby('label').aggregate('first').iterrows(), total=300):
    subprocess.call([
        'convert', '/m/nbe/archive/redness2/presentation/Presentation-Redness2-2015-06-29/stim_final20150610/%s' % meta.filename,
        '-resize', '128x128^',
        '-gravity', 'center',
        '-extent', '128x128',
        '/l/vanvlm1/redness2/images/%d.JPEG' % label
    ])
