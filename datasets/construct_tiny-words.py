"""
Construct a dataset containing 64x64 pixel images of rendered words.  Uses the
180 words used in the pilot study + 20 other frequently used Finnish words.

Last run as:
    python construct_word_stimuli.pi data/datasets/tiny-words
"""
# encoding: utf-8
import argparse
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib import font_manager as fm
import pandas as pd
from os import makedirs
from tqdm import tqdm
import pickle
from PIL import Image
from io import BytesIO

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate the tiny-words dataset')
parser.add_argument('path', type=str, help='The path to write the dataset to.')
parser.add_argument('set', type=str, help='Specify either "train" to generate the training set and to "test" to generate the test set.')
args = parser.parse_args()

# Limits
rotations = np.linspace(-20, 20, 11)
sizes = np.linspace(7, 16, 21)
fonts = ['courier new', 'dejavu sans mono', 'times new roman', 'arial',
         'arial black', 'verdana', 'comic sans ms', 'georgia',
         'liberation serif', 'impact', 'roboto condensed']
noise_levels = [0.2, 0.35, 0.5]

fonts = {
    'ubuntu mono': [None, '../data/fonts/UbuntuMono-R.ttf'],
    'courier': [None, '../data/fonts/courier.ttf'],
    'luxi mono regular': [None, '../data/fonts/luximr.ttf'],
    'lucida console': [None, '../data/fonts/LucidaConsole-R.ttf'],
    'lekton': [None, '../data/fonts/Lekton-Regular.ttf'],
    'dejavu sans mono': ['dejavu sans mono', None],
    'times new roman': [None, '../data/fonts/times.ttf'],
    'arial': [None, '../data/fonts/arial.ttf'],
    'arial black': [None, '../data/fonts/arialbd.ttf'],
    'verdana': [None, '../data/fonts/verdana.ttf'],
    'comic sans ms': [None, '../data/fonts/comic.ttf'],
    'georgia': [None, '../data/fonts/georgia.ttf'],
    'liberation serif': ['liberation serif', None],
    'impact': [None, '../data/fonts/impact.ttf'],
    'roboto condensed': [None, '../data/fonts/Roboto-Light.ttf'],
}

# Use the pilot stimulus list to select words to plot
stimuli = pd.read_csv('../data/pilot_data/pilot2/run1.csv', index_col=0).query('type=="word"')
words = stimuli['text']

# Add some common finnish words to pad the list to 200
more_words = pd.read_csv('../data/parsebank_v4_ud_scrambled.wordlist.txt', sep=' ', nrows=500, quoting=3, usecols=[1], header=None)
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
chosen_noise_levels = []

n = 500 if args.set == 'train' else 50
data = []
labels = np.zeros(len(words) * n, dtype=np.int)

dpi = 96.
f = Figure(figsize=(64 / dpi, 64 / dpi), dpi=dpi)
canvas = FigureCanvasAgg(f)
for label, word in tqdm(enumerate(words), total=len(words)):
    for i in range(n):
        f.clf()
        ax = f.add_axes([0, 0, 1, 1])
        rotation = rng.choice(rotations)
        fontsize = rng.choice(sizes)
        font = rng.choice(list(fonts.keys()))
        fontfamily, fontfile = fonts[font]
        fontprop = fm.FontProperties(family=fontfamily, fname=fontfile)
        noise_level = rng.choice(noise_levels)
        noise = rng.rand(64, 64)
        ax.imshow(noise, extent=[0, 1, 0, 1], cmap='gray', alpha=noise_level)
        ax.text(0.5, 0.5, word, ha='center', va='center',
                rotation=rotation, fontsize=fontsize, fontproperties=fontprop, alpha=1 - noise_level)
        #ax.text(0.5, 0.5, word, ha='center', va='center',
        #        rotation=rotation, fontsize=fontsize, fontproperties=fontprop)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        #plt.savefig(path + '/%03d.JPEG' % i)
        canvas.draw()
        buffer, (width, height) = canvas.print_to_buffer()
        image = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))[:, :, :3]

        chosen_words.append(word)
        chosen_rotations.append(rotation)
        chosen_sizes.append(fontsize)
        chosen_fonts.append(font)
        chosen_noise_levels.append(noise_level)

        buf = BytesIO()
        Image.fromarray(image.astype(np.uint8)).save(buf, format='png')
        img_compressed = buf.getvalue()

        #data[label * n + i] = image
        data.append(img_compressed)
        labels[label * n + i] = label

df = pd.DataFrame(dict(word=chosen_words, rotation=chosen_rotations,
                       size=chosen_sizes, font=chosen_fonts, label=labels,
                       noise_level=chosen_noise_levels))


makedirs(args.path, exist_ok=True)

df.to_csv(f'{args.path}/{args.set}.csv')
with open(f'{args.path}/{args.set}', 'wb') as f:
    pickle.dump(dict(data=data, labels=labels), f)
