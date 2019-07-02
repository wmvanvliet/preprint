# encoding: utf-8
import argparse
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from os import makedirs
import os.path as op
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate text stimuli')
parser.add_argument('path', type=str, help='The path to write the dataset to.')
parser.add_argument('set', type=str, help='Specify either "train" to generate the training set and to "test" to generate the test set.')
parser.add_argument('words', type=str, help='Specify either "train" to generate the training set and to "test" to generate the test set.')
args = parser.parse_args()

# Limits
rotations = np.linspace(-20, 20, 11)
sizes = np.linspace(10, 28, 21)
fonts = ['courier new', 'dejavu sans mono', 'times new roman', 'arial',
         'arial black', 'verdana', 'comic sans ms', 'georgia',
         'liberation serif', 'impact', 'roboto condensed']

# Use the SUBTLEX-US database to select words to plot
subtlex_us = pd.read_table('subtlex-us.txt', index_col=0)

# Select words between 2 and 6 letters long
subtlex_us = subtlex_us[subtlex_us.index.str.len() <= 6]
subtlex_us = subtlex_us[subtlex_us.index.str.len() >= 2]

# Drop words with capitals (like names)
subtlex_us = subtlex_us[subtlex_us.index.str.lower() == subtlex_us.index]

# Use top 1000 most frequent words
subtlex_us.sort_values('FREQcount', inplace=True, ascending=False)
words = subtlex_us.head(1000).index.values

rng = np.random.RandomState(0)

chosen_rotations = []
chosen_sizes = []
chosen_fonts = []
chosen_words = []

plt.close('all')
for word in tqdm(words, total=len(words)):
    path = op.join(args.path, word)
    makedirs(path)
    n = 200 if args.set == 'train' else 50
    for i in range(n):
        dpi = 96.
        f = plt.figure(figsize=(128 / dpi, 128 / dpi), dpi=dpi)
        ax = f.add_axes([0, 0, 1, 1])

        rotation = rng.choice(rotations)
        fontsize = rng.choice(sizes)
        family = rng.choice(fonts)
        ax.text(0.5, 0.5, word, ha='center', va='center',
                rotation=rotation, fontsize=fontsize, family=family)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.savefig(path + '/%03d.JPEG' % i)
        plt.close()

        chosen_words.append(word)
        chosen_rotations.append(rotation)
        chosen_sizes.append(fontsize)
        chosen_fonts.append(family)

df = pd.DataFrame(dict(word=chosen_words, rotation=chosen_rotations,
                       size=chosen_sizes, font=chosen_fonts))
df.to_csv('word_stimuli_%s.csv' % args.set)
