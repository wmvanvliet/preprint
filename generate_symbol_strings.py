# encoding: utf-8
import argparse
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from os import makedirs
import os.path as op
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate symbol string stimuli')
parser.add_argument('path', type=str, help='The path to write the dataset to.')
args = parser.parse_args()

num_strings = 200  # Number of strings to generate

# Limits
lengths = [2, 3, 4, 5, 6]
rotations = np.linspace(-20, 20, 11)
sizes = np.linspace(8, 14, 21)
fonts = ['courier new', 'dejavu sans mono', 'times new roman', 'arial',
         'arial black', 'verdana', 'comic sans ms', 'georgia',
         'liberation serif', 'impact', 'roboto condensed']
symbols = [
    b'\u25A1',  # square
    b'\u25EF',  # circle
    b'\u25B3',  # triangle
    b'\u25C7',  # diamond
]
          
rng = np.random.RandomState(0)

chosen_rotations = []
chosen_sizes = []
chosen_fonts = []
chosen_strings = []

plt.close('all')
pbar = tqdm(total=num_strings)
for i in range(num_strings):
    string_length = rng.choice(lengths)
    string = b''
    for j in range(string_length):
        string += rng.choice(symbols)
    string = string.decode('unicode-escape')

    path = op.join(args.path, '%03d' % i)
    makedirs(path)
    n = 50
    for i in range(n):
        dpi = 96.
        f = plt.figure(figsize=(64 / dpi, 64 / dpi), dpi=dpi)
        ax = f.add_axes([0, 0, 1, 1])

        rotation = rng.choice(rotations)
        fontsize = rng.choice(sizes)
        #family = rng.choice(fonts)
        family = 'dejavu sans mono'
        ax.text(0.5, 0.5, string, ha='center', va='center',
                rotation=rotation, fontsize=fontsize, family=family)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.savefig(path + '/%03d.JPEG' % i)
        plt.close()

        chosen_strings.append(string)
        chosen_rotations.append(rotation)
        chosen_sizes.append(fontsize)
        chosen_fonts.append(family)
    pbar.update(1)

df = pd.DataFrame(dict(word=chosen_strings, rotation=chosen_rotations,
                       size=chosen_sizes, font=chosen_fonts))
df.to_csv('symbol_string_stimuli.csv')
