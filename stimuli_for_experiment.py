"""
3 blocks of 15 minutes each. 15 * 60 = 600 seconds
180 Word trials of 1.5 seconds: 270 seconds
90 Consonant string trials of 1.5 seconds: 135 seconds
90 Symbol string trials of 1.5 seconds: 135 seconds
60 Question trials of 1 second: 60 seconds

1050 trials in total
540 words
270 consonant strings
270 symbol strings
180 question trials

stimulus length: 3-5 letters

question: is the given symbol in the correct location?
stimulus: KOIRA
question: _ O _ _ _ (correct)
          _ _ _ M _ (incorrect)
"""
#encoding: utf8
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import itertools
from scipy.io import loadmat

# Use a very specific random generator with a specific seed so in the future,
# this script will still generate the same stimuli.
rng = np.random.Generator(np.random.PCG64(18420))

# Load word2vec words. Filter for some desired properties
m = loadmat('word2vec.mat')
sel = np.array([np.all([l in 'abcdefghijklmnopqrstuvwxyzäöü' for l in word.strip()] + [3 <= len(word.strip()) <= 5])
                for word in m['vocab']])
vocab = [word.strip() for word in m['vocab'][sel]]
vocab_sel_3 = np.flatnonzero([len(word) == 3 for word in vocab])
vocab_sel_4 = np.flatnonzero([len(word) == 4 for word in vocab])
vocab_sel_5 = np.flatnonzero([len(word) == 5 for word in vocab])
freqs = m['freqs'][0, sel]
vectors = m['vectors'][sel]

symbols = {
    's': '\u25FB', # Square
    'o': '\u25CB', # Circle
    '^': '\u25B3', # Triangle up
    'v': '\u25BD', # Triangle down
    'd': '\u25C7', # Diamond
}
consonants = list('bcdfghjklmnpqrstvwxz')

rotations = [0, +15]
fontsizes = [15, 30]
noise_levels = [0.2, 0.5]
fonts = ['Comic Sans MS', 'Times New Roman']

n_word_strings = 180
word_strings = [vocab[i] for i in rng.choice(vocab_sel_3, 60)]
word_strings += [vocab[i] for i in rng.choice(vocab_sel_4, 60)]
word_strings += [vocab[i] for i in rng.choice(vocab_sel_5, 60)]
rng.shuffle(word_strings)

n_symbol_strings = 90
symbol_strings = [''.join(rng.choice(list(symbols.keys()), 3)) for _ in range(30)]
symbol_strings += [''.join(rng.choice(list(symbols.keys()), 4)) for _ in range(30)]
symbol_strings += [''.join(rng.choice(list(symbols.keys()), 5)) for _ in range(30)]
rng.shuffle(symbol_strings)

n_consonsant_strings = 90
consonant_strings = [''.join(rng.choice(consonants, 3)) for _ in range(30)]
consonant_strings += [''.join(rng.choice(consonants, 4)) for _ in range(30)]
consonant_strings += [''.join(rng.choice(consonants, 5)) for _ in range(30)]
rng.shuffle(consonant_strings)

types = (['word'] * n_word_strings) + (['symbols'] * n_symbol_strings) + (['consonants'] * n_consonsant_strings)

stimuli = []
stimuli_iter = zip(
    types,
    word_strings + symbol_strings + consonant_strings,
    itertools.cycle(itertools.product(fonts, fontsizes, rotations, noise_levels)),
)
for type, text, (font, fontsize, rotation, noise_level) in stimuli_iter:
    if type == 'symbols':
        font = 'DejaVu Sans'  # Symbols only render correctly in this font
    stimuli.append(dict(type=type, text=text, font=font, fontsize=fontsize, rotation=rotation, noise_level=noise_level))
rng.shuffle(stimuli)
stimuli = pd.DataFrame(stimuli)

# Create image files
plt.close('all')
dpi = 96.
width, height = 800, 400
f = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

# Fixation cross
plt.clf()
ax = f.add_axes([0, 0, 1, 1])
ax.set_facecolor((0.5, 0.5, 0.5))
plt.plot([0.49, 0.51], [0.5, 0.5], color='black', linewidth=1)
plt.plot([0.5, 0.5], [0.48, 0.52], color='black', linewidth=1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('data/presentation/stimuli/fixation_cross.JPEG')

def text_stimulus(type, text, font='DejaVu Sans', fontsize=30, rotation=0, noise_level=0):
    plt.clf()
    ax = f.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.5, 0.5, 0.5))

    noise_level = noise_level
    noise = np.random.rand(width, height)
    ax.imshow(noise, extent=[0, 1, 0, 1], cmap='gray', alpha=noise_level, aspect='auto')

    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=fontsize,
            family=font, alpha=1 - noise_level, rotation=rotation)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(f'data/presentation/stimuli/{type}_{text}.JPEG')

# Stimuli
for i, stimulus in tqdm(stimuli.iterrows(), total=len(stimuli)):
    if stimulus['type'] == 'symbols':
        for k, v in symbols.items():
            stimulus['text'] = stimulus['text'].replace(k, v)
    text_stimulus(**stimulus)

plt.close()
