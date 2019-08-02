# encoding: utf-8
import argparse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
import pandas as pd
from os import makedirs
import os.path as op
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate text stimuli')
parser.add_argument('path', type=str, help='The path to write the dataset to.')
parser.add_argument('set', type=str, help='Specify either "train" to generate the training set and to "test" to generate the test set.')
args = parser.parse_args()

# Limits
rotations = np.linspace(-20, 20, 11)
sizes = np.linspace(7, 16, 21)
fonts = ['courier new', 'dejavu sans mono', 'times new roman', 'arial',
         'arial black', 'verdana', 'comic sans ms', 'georgia',
         'liberation serif', 'impact', 'roboto condensed']

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

# Use the redness1 list to select words to plot
stimuli = pd.read_csv('/l/vanvlm1/redness1/stimuli.csv')
words = stimuli['ITEM']

# Add some common finnish words to pad the list to 200
more_words = pd.read_csv('/m/nbe/project/corpora/FinnishParseBank/parsebank_v4_ud_scrambled.wordlist.txt', sep=' ', nrows=500, quoting=3, usecols=[1], header=None)
more_words.columns = ['ITEM']

# Select words between 2 and 6 letters long
more_words = more_words[more_words.ITEM.str.len() <= 6]
more_words = more_words[more_words.ITEM.str.len() >= 2]

# Drop words with capitals (like names)
more_words = more_words[more_words.ITEM.str.lower() == more_words.ITEM]

# Drop punctuation
more_words = more_words[~more_words.ITEM.str.contains('.', regex=False) & ~more_words.ITEM.str.contains(',', regex=False)]

# Pad the original word list up to 200 words
words = pd.concat([words, more_words['ITEM']], ignore_index=True)
words = words.drop_duplicates()
words = words[:200]

rng = np.random.RandomState(0)

chosen_rotations = []
chosen_sizes = []
chosen_fonts = []
chosen_words = []

plt.close('all')
dpi = 96.
f = plt.figure(figsize=(64 / dpi, 64 / dpi), dpi=dpi)
for word in tqdm(words, total=len(words)):
    path = op.join(args.path, word)
    makedirs(path)
    n = 200 if args.set == 'train' else 50
    for i in range(n):
        plt.clf()
        ax = f.add_axes([0, 0, 1, 1])
        rotation = rng.choice(rotations)
        fontsize = rng.choice(sizes)
        font = rng.choice(list(fonts.keys()))
        fontfamily, fontfile = fonts[font]
        fontprop = fm.FontProperties(family=fontfamily, fname=fontfile)
        ax.text(0.5, 0.5, word, ha='center', va='center',
                rotation=rotation, fontsize=fontsize, fontproperties=fontprop)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.savefig(path + '/%03d.JPEG' % i)

        chosen_words.append(word)
        chosen_rotations.append(rotation)
        chosen_sizes.append(fontsize)
        chosen_fonts.append(font)
plt.close()

df = pd.DataFrame(dict(word=chosen_words, rotation=chosen_rotations,
                       size=chosen_sizes, font=chosen_fonts))
df.to_csv('word_stimuli_%s.csv' % args.set)
