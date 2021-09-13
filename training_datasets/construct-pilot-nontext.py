"""
Construct a dataset containing 64x64 pixel images of rendered mixed consonant
and symbols.

Last run as:
    python construct_pilot-nontext.py data/datasets/pilot-nontext
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
from PIL import Image
from io import BytesIO
import tfrecord

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate the pilot-nontext dataset')
parser.add_argument('path', type=str, help='The path to write the dataset to.')
parser.add_argument('set', type=str, help='Specify either "train" to generate the training set and to "test" to generate the test set.')
args = parser.parse_args()

# Set this to wherever /m/nbe/scratch/reading_models is mounted on your system
data_path = '/m/nbe/scratch/reading_models'

consonants = list('bcdfghjklmnpqrstvwxz')
symbols = [
    '\u25A1',  # square
    '\u25EF',  # circle
    '\u25B3',  # triangle
    '\u25C7',  # diamond
]
alphabet = consonants + symbols

# Limits
rotations = np.linspace(-20, 20, 11)
sizes = np.linspace(14, 32, 42)
fonts = ['courier new', 'dejavu sans mono', 'times new roman', 'arial',
         'arial black', 'verdana', 'comic sans ms', 'georgia',
         'liberation serif', 'impact', 'roboto condensed']
noise_levels = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
lengths = [2, 3, 4, 5, 6]

fonts = {
    'ubuntu mono': [None, '../fonts/UbuntuMono-R.ttf'],
    'courier': [None, '../fonts/courier.ttf'],
    'luxi mono regular': [None, '../fonts/luximr.ttf'],
    'lucida console': [None, '../fonts/LucidaConsole-R.ttf'],
    'lekton': [None, '../fonts/Lekton-Regular.ttf'],
    'dejavu sans mono': [None, '../fonts/DejaVuSansMono.ttf'],
    'times new roman': [None, '../fonts/times.ttf'],
    'arial': [None, '../fonts/arial.ttf'],
    'arial black': [None, '../fonts/arialbd.ttf'],
    'verdana': [None, '../fonts/verdana.ttf'],
    'comic sans ms': [None, '../fonts/comic.ttf'],
    'georgia': [None, '../fonts/georgia.ttf'],
    'liberation serif': [None, '../fonts/LiberationSerif-Regular.ttf'],
    'impact': [None, '../fonts/impact.ttf'],
    'roboto condensed': [None, '../fonts/Roboto-Light.ttf'],
}
rng = np.random.RandomState(0)

# Render the consonant strings as PNG files
chosen_rotations = []
chosen_sizes = []
chosen_fonts = []
chosen_strings = []
chosen_noise_levels = []
chosen_lengths = []

n = 100_000 if args.set == 'train' else 10_000
data = []
labels = np.zeros(n, dtype=np.int)

makedirs(args.path, exist_ok=True)
writer = tfrecord.TFRecordWriter(f'{args.path}/{args.set}.tfrecord')

dpi = 96.
width = 256  # Pixels
height = 256  # Pixels
f = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
canvas = FigureCanvasAgg(f)
for i in tqdm(range(n), total=n):
    f.clf()
    ax = f.add_axes([0, 0, 1, 1])
    rotation = rng.choice(rotations)
    fontsize = rng.choice(sizes)
    font = rng.choice(list(fonts.keys()))
    fontfamily, fontfile = fonts[font]
    fontprop = fm.FontProperties(family=fontfamily, fname=fontfile, size=fontsize)
    noise_level = rng.choice(noise_levels)
    length = rng.choice(lengths)
    noise = rng.random((width, height))
    ax.imshow(noise, extent=[0, 1, 0, 1], cmap='gray', alpha=noise_level)

    # Generate the consonant string
    text = ''.join(rng.choice(alphabet, length))
    ax.text(0.5, 0.5, text, ha='center', va='center',
            rotation=rotation, fontproperties=fontprop, alpha=1 - noise_level)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    canvas.draw()
    buffer, (width, height) = canvas.print_to_buffer()
    image = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))[:, :, :3]

    chosen_strings.append(text)
    chosen_rotations.append(rotation)
    chosen_sizes.append(fontsize)
    chosen_fonts.append(font)
    chosen_noise_levels.append(noise_level)
    chosen_lengths.append(length)

    buf = BytesIO()
    Image.fromarray(image.astype(np.uint8)).save(buf, format='jpeg')

    writer.write({
        'image/height': (height, 'int'),
        'image/width': (width, 'int'),
        'image/colorspace': (b'RGB', 'byte'),
        'image/channels': (3, 'int'),
        'image/class/label': (0, 'int'),
        'image/class/text': (b'nontext', 'byte'),
        'image/format': (b'JPEG', 'byte'),
        'image/filename': (f'nontext-{i:03d}.jpg'.encode('utf-8'), 'byte'),
        'image/encoded': (buf.getvalue(), 'byte'),
    })

tfrecord.tools.create_index(f'{args.path}/{args.set}.tfrecord', f'{args.set}.index')

df = pd.DataFrame(dict(text=chosen_strings, rotation=chosen_rotations,
                       size=chosen_sizes, font=chosen_fonts, label=labels,
                       noise_level=chosen_noise_levels, lenght=chosen_lengths))
df.to_csv(f'{args.path}/{args.set}.csv')
pd.DataFrame(np.zeros((1, 300)), index=[0]).to_csv(f'{args.path}/vectors.csv')
