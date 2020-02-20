"""
Construct a dataset containing 64x64 pixel images of rendered consonant
strings. Uses the 90 words used in the pilot study + 110 other randomly
renerated ones.

Last run as:
    python construct_tiny-consinants.pi data/datasets/tiny-consonants
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
parser = argparse.ArgumentParser(description='Generate the tiny-consonants dataset')
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

# Use the pilot stimulus list to select the first 90 consonants strings
stimuli = pd.read_csv('../data/pilot_data/pilot2/run1.csv', index_col=0).query('type=="consonants"')
consonant_strings = stimuli['text'].values.tolist()

# Generate 110 more strings to bring it to 200 in total
consonants = list('bcdfghjklmnpqrstvwxz')

# Use a very specific random generator with a specific seed so in the future,
# this script will still generate the same stimuli.
rng = np.random.Generator(np.random.PCG64(33921))

more_consonant_strings = list(set(''.join(rng.choice(consonants, 4)) for _ in range(80)))[:40]
more_consonant_strings += list(set(''.join(rng.choice(consonants, 5)) for _ in range(80)))[:40]
more_consonant_strings += list(set(''.join(rng.choice(consonants, 6)) for _ in range(60)))[:30]
rng.shuffle(more_consonant_strings)

# Pad the original consonant strings list to 200 words
consonant_strings += more_consonant_strings

# Render the consonant strings as PNG files
chosen_rotations = []
chosen_sizes = []
chosen_fonts = []
chosen_words = []
chosen_noise_levels = []

n = 500 if args.set == 'train' else 50
data = []
labels = np.zeros(len(consonant_strings) * n, dtype=np.int)

dpi = 96.
f = Figure(figsize=(64 / dpi, 64 / dpi), dpi=dpi)
canvas = FigureCanvasAgg(f)
for label, text in tqdm(enumerate(consonant_strings), total=len(consonant_strings)):
    for i in range(n):
        f.clf()
        ax = f.add_axes([0, 0, 1, 1])
        rotation = rng.choice(rotations)
        fontsize = rng.choice(sizes)
        font = rng.choice(list(fonts.keys()))
        fontfamily, fontfile = fonts[font]
        fontprop = fm.FontProperties(family=fontfamily, fname=fontfile)
        noise_level = rng.choice(noise_levels)
        noise = rng.random((64, 64))
        ax.imshow(noise, extent=[0, 1, 0, 1], cmap='gray', alpha=noise_level)
        ax.text(0.5, 0.5, text, ha='center', va='center',
                rotation=rotation, fontsize=fontsize, fontproperties=fontprop, alpha=1 - noise_level)
        #ax.text(0.5, 0.5, text, ha='center', va='center',
        #        rotation=rotation, fontsize=fontsize, fontproperties=fontprop)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        #plt.savefig(path + '/%03d.JPEG' % i)
        canvas.draw()
        buffer, (width, height) = canvas.print_to_buffer()
        image = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))[:, :, :3]

        chosen_words.append(text)
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

df = pd.DataFrame(dict(text=chosen_words, rotation=chosen_rotations,
                       size=chosen_sizes, font=chosen_fonts, label=labels,
                       noise_level=chosen_noise_levels))


makedirs(args.path, exist_ok=True)

df.to_csv(f'{args.path}/{args.set}.csv')
with open(f'{args.path}/{args.set}', 'wb') as f:
    pickle.dump(dict(data=data, labels=labels), f)

with open(f'{args.path}/meta', 'wb') as f:
    pickle.dump(dict(label_names=chosen_words), f)
