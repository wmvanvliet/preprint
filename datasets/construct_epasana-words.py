"""
Construct a dataset containing 128x128 pixel images of rendered words.  Uses the
118 words used in the "epasana" study + 82 other frequently used Finnish words.
"""
# encoding: utf-8
import argparse
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib import font_manager as fm
import pandas as pd
from scipy.io import loadmat
from os import makedirs
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import tfrecord

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
words = pd.read_csv(f'/m/nbe/scratch/epasana/stimuli.csv').query('type=="word"')['text']

# Adding some common finnish words to pad the list to 200
more_words = pd.read_csv(f'{data_path}/parsebank_v4_ud_scrambled.wordlist.txt', sep=' ', nrows=500, quoting=3, usecols=[1], header=None)
more_words.columns = ['ITEM']

# Selecting words between 7 and 8 letters long
more_words = more_words[more_words.ITEM.str.len() <= 8]
more_words = more_words[more_words.ITEM.str.len() >= 7]

# Drop words with capitals (like names)
more_words = more_words[more_words.ITEM.str.lower() == more_words.ITEM]

# Drop punctuation
more_words = more_words[~more_words.ITEM.str.contains('.', regex=False) & ~more_words.ITEM.str.contains(',', regex=False) & ~more_words.ITEM.str.contains('#', regex=False)]
more_words = [x.upper() for x in more_words['ITEM']]
more_words = pd.DataFrame(more_words)

# Pad the original word list up to 200 words
words = pd.concat([words, more_words], ignore_index=True)
words = words.drop_duplicates()
words = words[:200]
words  = words[0]

# Get word2vec vectors for the words
word2vec = loadmat(f'{data_path}/word2vec.mat')
vocab = [v.strip() for v in word2vec['vocab']]

vectors = []
for w in words:
    w = w.lower()

    # Some words need translation to their lemma form
    if w == 'maalari':
        w = 'taide#maalari'
    elif w == 'luominen':
        w = 'luomus'
    elif w == 'oleminen':
        w = 'olemus'
    elif w == 'eläminen':
        w = 'elatus'
    elif w == 'koraani':
        w = 'koraanin'
    vectors.append(word2vec['vectors'][vocab.index(w)])

rng = np.random.RandomState(0)

chosen_rotations = []
chosen_sizes = []
chosen_fonts = []
chosen_words = []

n = 500 if args.set == 'train' else 50
labels = np.zeros(len(words) * n, dtype=np.int)

makedirs(args.path, exist_ok=True)
writer = tfrecord.TFRecordWriter(f'{args.path}/{args.set}.tfrecord')

dpi = 96.
f = Figure(figsize=(256 / dpi, 256 / dpi), dpi=dpi)
canvas = FigureCanvasAgg(f)
for label, word in tqdm(enumerate(words), total=len(words)):
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

        #f.savefig('training_imgs/imgs/'+word+'%03d.JPEG' % i)

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

tfrecord.tools.create_index(f'{args.path}/{args.set}.tfrecord', f'{args.set}.index')

df = pd.DataFrame(dict(text=chosen_words, rotation=chosen_rotations,
                       size=chosen_sizes, font=chosen_fonts, label=labels))
df.to_csv(f'{args.path}/{args.set}.csv')
pd.DataFrame(vectors, index=words).to_csv(f'{args.path}/vectors.csv')
