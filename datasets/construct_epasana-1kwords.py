# encoding: utf-8
"""
Construct a dataset containing 256x256 pixel images of rendered words. Uses the
118 words used in the "epasana" study + 82 other frequently used Finnish words.
"""
import argparse
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib import font_manager as fm
import pandas as pd
from os import makedirs
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import tfrecord
from gensim.models import KeyedVectors
import re

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate the epasana-words dataset')
parser.add_argument('path', type=str, help='The path to write the dataset to.')
parser.add_argument('set', type=str, help='Specify either "train" to generate the training set and to "test" to generate the test set.')
args = parser.parse_args()

# Set this to wherever /m/nbe/scratch/reading_models is mounted on your system
data_path = '/m/nbe/scratch/reading_models'

# Limits
rotations = np.linspace(-20, 20, 11)
sizes = np.linspace(14, 32, 42)
fonts = ['courier new', 'dejavu sans mono', 'times new roman', 'arial',
         'arial black', 'verdana', 'comic sans ms', 'georgia',
         'liberation serif', 'impact', 'roboto condensed']

fonts = {
    'ubuntu mono': [None, f'{data_path}/fonts/UbuntuMono-R.ttf'],
    'courier': [None, f'{data_path}/fonts/courier.ttf'],
    'luxi mono regular': [None, f'{data_path}/fonts/luximr.ttf'],
    'lucida console': [None, f'{data_path}/fonts/LucidaConsole-R.ttf'],
    'lekton': [None, f'{data_path}/fonts/Lekton-Regular.ttf'],
    'dejavu sans mono': [None, f'{data_path}/fonts/DejaVuSansMono.ttf'],
    'times new roman': [None, f'{data_path}/fonts/times.ttf'],
    'arial': [None, f'{data_path}/fonts/arial.ttf'],
    'arial black': [None, f'{data_path}/fonts/arialbd.ttf'],
    'verdana': [None, f'{data_path}/fonts/verdana.ttf'],
    'comic sans ms': [None, f'{data_path}/fonts/comic.ttf'],
    'georgia': [None, f'{data_path}/fonts/georgia.ttf'],
    'liberation serif': [None, f'{data_path}/fonts/LiberationSerif-Regular.ttf'],
    'impact': [None, f'{data_path}/fonts/impact.ttf'],
    'roboto condensed': [None, f'{data_path}/fonts/Roboto-Light.ttf'],
}

# Using the "epasana" stimulus list to select words to plot
words = pd.read_csv('/m/nbe/scratch/epasana/stimuli.csv').query('type=="word"')['text']
words = words.str.lower()

# Read the Finnish Parsebank language model
print('Reading Finnish Parsebank...', flush=True, end='')
vectors = KeyedVectors.load_word2vec_format('/m/nbe/project/corpora/big/parsebank_v4/finnish_parsebank_v4_lemma_5+5.bin', binary=True)
print('done.')

# Adding some common finnish words to pad the list to 1_000
more_words = list(vectors.vocab.keys())[:1_000_000]

# Drop words containing capitals (like names) and punctuation characters
pattern = re.compile('^[a-zäö#]+$')
more_words = [w for w in more_words if pattern.match(w)]

# Do we have enough words left after our filters?
assert len(more_words) >= 1_000

# Pad the original word list up to 1_000 words
more_words = pd.DataFrame(more_words)
words = pd.concat([words, more_words], ignore_index=True)
words = words.drop_duplicates()
words = list(words[:1_000][0])

# Fix lemmatization of some words
for i, w in enumerate(words):
    if w == 'maalari':
        words[i] = 'taide#maalari'
    elif w == 'luominen':
        words[i] = 'luomus'
    elif w == 'oleminen':
        words[i] = 'olemus'
    elif w == 'eläminen':
        words[i] = 'elatus'
    elif w == 'koraani':
        words[i] = 'koraanin'

# Perform a lookup for the w2v vectors for each chosen word
vectors = vectors[words]

# Start generating images
rng = np.random.RandomState(0)

chosen_rotations = []
chosen_sizes = []
chosen_fonts = []
chosen_words = []

n = 100 if args.set == 'train' else 10
labels = np.zeros(len(words) * n, dtype=np.int)

makedirs(args.path, exist_ok=True)
writer = tfrecord.TFRecordWriter(f'{args.path}/{args.set}.tfrecord')

dpi = 96.
f = Figure(figsize=(256 / dpi, 256 / dpi), dpi=dpi)
canvas = FigureCanvasAgg(f)
for label, word in tqdm(enumerate(words), total=len(words)):
    word = word.replace('#', '')
    for i in range(n):
        f.clf()
        ax = f.add_axes([0, 0, 1, 1])
        rotation = rng.choice(rotations)
        fontsize = rng.choice(sizes)
        font = rng.choice(list(fonts.keys()))
        fontfamily, fontfile = fonts[font]
        fontprop = fm.FontProperties(family=fontfamily, fname=fontfile, size=fontsize)
        ax.text(0.5, 0.5, word, ha='center', va='center', rotation=rotation,
                fontproperties=fontprop)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()

        canvas.draw()
        buffer, (width, height) = canvas.print_to_buffer()
        image = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))[:, :, :3]

        chosen_words.append(word)
        chosen_rotations.append(rotation)
        chosen_sizes.append(fontsize)
        chosen_fonts.append(font)

        buf = BytesIO()
        Image.fromarray(image.astype(np.uint8)).save(buf, format='jpeg')

        writer.write({
            'image/height': (height, 'int'),
            'image/width': (width, 'int'),
            'image/colorspace': (b'RGB', 'byte'),
            'image/channels': (3, 'int'),
            'image/class/label': (label, 'int'),
            'image/class/text': (word.encode('utf-8'), 'byte'),
            'image/format': (b'JPEG', 'byte'),
            'image/filename': (f'{word}-{i:03d}.jpg'.encode('utf-8'), 'byte'),
            'image/encoded': (buf.getvalue(), 'byte'),
        })

        labels[label * n + i] = label
writer.close()

tfrecord.tools.create_index(f'{args.path}/{args.set}.tfrecord', f'{args.path}/{args.set}.index')

df = pd.DataFrame(dict(text=chosen_words, rotation=chosen_rotations,
                       size=chosen_sizes, font=chosen_fonts, label=labels))
df.to_csv(f'{args.path}/{args.set}.csv')
pd.DataFrame(vectors, index=words).to_csv(f'{args.path}/vectors.csv')
