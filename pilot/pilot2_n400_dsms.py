import rsa
import mne
import pandas as pd
import numpy as np
import pickle
from scipy.io import loadmat
from matplotlib import pyplot as plt
from scipy.spatial import distance

epochs = mne.read_epochs('../data/pilot_data/pilot2/pilot2_epo.fif', preload=False)
epochs = epochs[['word', 'symbols', 'consonants']]
stimuli = epochs.metadata.groupby('text').agg('first').sort_values('event_id')
stimuli['y'] = np.arange(len(stimuli))
metadata = pd.merge(epochs.metadata, stimuli[['y']], left_on='text', right_index=True).sort_index()
assert np.array_equal(metadata.event_id.values.astype(int), epochs.events[:, 2])

# Get word2vec vectors for the words
word2vec = loadmat('../data/word2vec.mat')
vocab = [v.strip() for v in word2vec['vocab']]
word_vectors = []
non_word_vectors = []
for text, [type] in stimuli[['type']].iterrows():
    if type == 'word':
        #word_vectors.append(word2vec['vectors'][vocab.index(text)])
        vector = np.zeros(300)
        vector[np.random.choice(np.arange(300), 20)] = 1
        word_vectors.append(vector)
    else:
        vector = np.random.randn(300) * 1
        non_word_vectors.append(vector)
#word_vectors = np.array(word_vectors) - np.mean(word2vec['vectors'], axis=0)
word_vectors = np.array(word_vectors) # - np.mean(word_vectors, axis=0)
non_word_vectors = np.array(non_word_vectors)
vectors = np.vstack((word_vectors, non_word_vectors))

def words1(x, y):
    if (x != 'word' or y != 'word'):
        return 1
    else:
        return 0

def words2(x, y):
    if (x == 'word' and y == 'word'):
        return 0
    elif (x == 'consonants' and y == 'consonants'):
        return 0
    elif (x == 'symbols' and y == 'symbols'):
        return 0
    else:
        return 1

def words3(x, y):
    if (x == 'word' or y == 'word'):
        return 1
    else:
        return 0

def words4(x, y):
    if (x == 'word' and y == 'word'):
        return 0
    elif (x == 'consonants' and y == 'consonants'):
        return 0
    elif (x == 'symbols' and y == 'symbols'):
        return 0
    elif (x == 'consonants' and y == 'symbols'):
        return 0
    elif (x == 'symbols' and y == 'consonants'):
        return 0
    else:
        return 1

def letters1(x, y):
    if (x == 'symbols' or y == 'symbols'):
        return 1
    else:
        return 0

def letters2(x, y):
    if (x == 'symbols' or y == 'symbols') and (x != y):
        return 1
    else:
        return 0

print('Computing model DSMs...', end='', flush=True)
dsm_models = [
    rsa.compute_dsm(stimuli[['type']], metric=words1),
    rsa.compute_dsm(stimuli[['type']], metric=words2),
    rsa.compute_dsm(stimuli[['type']], metric=words3),
    rsa.compute_dsm(stimuli[['type']], metric=words4),
    rsa.compute_dsm(stimuli[['type']], metric=letters1),
    rsa.compute_dsm(stimuli[['type']], metric=letters2),
    rsa.compute_dsm(vectors, metric='correlation'),
]
dsm_names = ['Words1', 'Words2', 'Words3', 'Words4', 'Letters1', 'Letters2', 'Word2Vec']

with open(f'../data/dsms/pilot2_n400_dsms.pkl', 'wb') as f:
    pickle.dump(dict(dsms=dsm_models, dsm_names=dsm_names), f)

fig, ax = plt.subplots(1, 7, sharex='col', sharey='row', figsize=(10, 3))
for col in range(7):
    if col < len(dsm_models):
        ax[col].imshow(distance.squareform(dsm_models[col]), cmap='magma')
        ax[col].set_title(dsm_names[col])
plt.tight_layout()
plt.savefig(f'../figures/pilot2_dsms_n400.pdf')
