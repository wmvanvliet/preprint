import mne
import numpy as np
import pandas as pd
import unicodedata
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
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
bad_subjects = [10, 11, 18]
pbar = tqdm(total=20 * 3 * 3)
for subject in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
    if subject in bad_subjects:
        print('Skipping', subject)
        continue
    for session in [1, 2, 3]:
        for run in [1, 2, 3]:
            epochs_fname = '/m/nbe/project/redness2/bids/derivatives/meg-derivatives/sub-{subject:02d}/ses-{session:02d}/sub-{subject:02d}_ses-{session:02d}_task-naming_run-{run:02d}_clean-epo.fif'
            events_fname = '/m/nbe/project/redness2/bids/sub-{subject:02d}/ses-{session:02d}/meg/sub-{subject:02d}_ses-{session:02d}_task-naming_run-{run:02d}_events.tsv'
            e = mne.read_epochs(epochs_fname.format(subject=subject, session=session, run=run), preload=True)
            metadata = pd.read_csv(events_fname.format(subject=subject, session=session, run=run), sep='\t')
            e.metadata = metadata.iloc[np.flatnonzero([len(d) == 0 for d in e.drop_log])]
            e.decimate(3)
            e.pick_types(meg='grad')
            all_epochs.append(e)
            pbar.update(1)
all_epochs = mne.concatenate_epochs(all_epochs)
pbar.close()

# Compute class labels
ref = all_epochs.metadata.drop_duplicates()
ref = ref.sort_values(['modality', 'word', 'filename', 'font']).reset_index(drop=True)
index = dict()
for i, row in ref.iterrows():
    index[tuple(row[:4])] = i
all_epochs.metadata['label'] = [index[tuple(row[:4])] for _, row in all_epochs.metadata.iterrows()]

# Compute noise cov
# noise_cov = mne.compute_covariance(all_epochs, tmin=-0.2)
# noise_cov.save('/l/vanvlm1/redness2/noise-cov.fif')

# Save epochs
all_epochs.save('/l/vanvlm1/redness2/all-epo.fif', overwrite=True)

fonts = {
    'ubuntu mono': [None, '/u/45/vanvlm1/unix/.fonts/UbuntuMono-R.ttf'],
    'courier': ['courier new', None],
    'luxi mono regular': [None, '/u/45/vanvlm1/unix/.fonts/luximr.ttf'],
    'lucida console': [None, '/u/45/vanvlm1/unix/.fonts/LucidaConsole-R.ttf'],
    'lekton': [None, '/u/45/vanvlm1/unix/.fonts/Lekton-Regular.ttf'],
    'dejavu sans mono': ['dejavu sans mono', None],
    'times new roman': ['times new roman', None],
    'arial': ['arial', None],
    'arial black': ['arial black', None],
    'verdana': ['verdana', None],
    'comic sans ms': ['comic sans ms', None],
    'georgia': ['georgia', None],
    'liberation serif': ['liberation serif', None],
    'impact': ['impact', None],
    'roboto condensed': ['roboto condensed', None]
}

# Create image files
plt.close('all')
dpi = 96.
f = plt.figure(figsize=(64 / dpi, 64 / dpi), dpi=dpi)
for label, meta in tqdm(all_epochs.metadata.query('modality=="wri"').groupby('label').aggregate('first').iterrows(), total=300):
    plt.clf()
    ax = f.add_axes([0, 0, 1, 1])
    fontsize = 19
    fontfamily, fontfile = fonts[meta.font.lower()]
    fontprop = fm.FontProperties(family=fontfamily, fname=fontfile)
    ax.text(0.5, 0.5, meta.word, ha='center', va='center',
            fontsize=fontsize, fontproperties=fontprop)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.savefig('/l/vanvlm1/redness2/images/%d.JPEG' % label)
plt.close()

for label, meta in tqdm(all_epochs.metadata.query('modality=="pic"').groupby('label').aggregate('first').iterrows(), total=300):
    subprocess.call([
        'convert', '/m/nbe/archive/redness2/presentation/Presentation-Redness2-2015-06-29/stim_final20150610/%s' % meta.filename,
        '-resize', '64x64^',
        '-gravity', 'center',
        '-extent', '64x64',
        '/l/vanvlm1/redness2/images/%d.JPEG' % label
    ])
