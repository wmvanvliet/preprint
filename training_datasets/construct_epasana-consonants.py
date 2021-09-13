"""
Construct a dataset containing 128x128 pixel images of random consonant strings.  Uses the
consonants used in the "epasana" study.
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
parser = argparse.ArgumentParser(description='Generate the epasana-consonants dataset')
parser.add_argument('path', type=str, help='The path to write the dataset to.')
parser.add_argument('set', type=str, help='Specify either "train" to generate the training set and to "test" to generate the test set.')
args = parser.parse_args()

data_path = '/m/nbe/scratch/reading_models'

# In Finnish, many consonants are barely used (only for loan words).
consonants = list('BDFGHJKLMNPRSTV')

rotations = np.linspace(-20, 20, 11)
sizes = np.linspace(14, 32, 42)
fonts = ['courier new', 'dejavu sans mono', 'times new roman', 'arial',
         'arial black', 'verdana', 'comic sans ms', 'georgia',
         'liberation serif', 'impact', 'roboto condensed']
lengths = [3, 4, 5, 6, 7, 8]
noise_levels = [0.1, 0.2, 0.3]

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

rng = np.random.RandomState(0)

# Render the consonant strings as PNG files
chosen_rotations = []
chosen_sizes = []
chosen_fonts = []
chosen_strings = []
chosen_lengths = []
chosen_noise_levels = []

n = 50_000 if args.set == 'train' else 5_000

makedirs(args.path, exist_ok=True)
writer = tfrecord.TFRecordWriter(f'{args.path}/{args.set}.tfrecord')

dpi = 96.
f = Figure(figsize=(256 / dpi, 256 / dpi), dpi=dpi)
canvas = FigureCanvasAgg(f)
for i in tqdm(range(n), total=n):
    f.clf()
    ax = f.add_axes([0, 0, 1, 1], label='background')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.fill_between([0, 1], [1, 1], color='#696969')

    rotation = rng.choice(rotations)
    fontsize = rng.choice(sizes)
    font = rng.choice(list(fonts.keys()))
    length = rng.choice(lengths)
    noise_level = rng.choice(noise_levels)

    # Generate the consonant string
    ax = f.add_axes([0, 0, 1, 1], label='text')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fontfamily, fontfile = fonts[font]
    fontprop = fm.FontProperties(family=fontfamily, fname=fontfile, size=fontsize)
    text = ''.join(rng.choice(consonants, length))
    ax.text(0.5, 0.5, text, ha='center', va='center', rotation=rotation,
            fontproperties=fontprop, alpha=1 - noise_level)

    ax = f.add_axes([0, 0, 1, 1], label='noise')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    noise = rng.rand(256, 256)
    ax.imshow(noise, extent=[0, 1, 0, 1], cmap='gray', alpha=noise_level)

    canvas.draw()
    buffer, (width, height) = canvas.print_to_buffer()
    image = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))[:, :, :3]

    chosen_strings.append(text)
    chosen_rotations.append(rotation)
    chosen_sizes.append(fontsize)
    chosen_fonts.append(font)
    chosen_lengths.append(length)
    chosen_noise_levels.append(noise_level)

    buf = BytesIO()
    Image.fromarray(image.astype(np.uint8)).save(buf, format='jpeg')
    #Image.fromarray(image.astype(np.uint8)).save('/tmp/test.jpg', format='jpeg')

    writer.write({
        'image/height': (height, 'int'),
        'image/width': (width, 'int'),
        'image/colorspace': (b'RGB', 'byte'),
        'image/channels': (3, 'int'),
        'image/class/label': (0, 'int'),
        'image/format': (b'JPEG', 'byte'),
        'image/filename': (f'{i:06d}.jpg'.encode('utf-8'), 'byte'),
        'image/encoded': (buf.getvalue(), 'byte'),
    })
writer.close()

tfrecord.tools.create_index(f'{args.path}/{args.set}.tfrecord', f'{args.path}/{args.set}.index')

makedirs(args.path, exist_ok=True)
df = pd.DataFrame(dict(text=chosen_strings, rotation=chosen_rotations, noise=chosen_noise_levels,
                       size=chosen_sizes, font=chosen_fonts, label=np.zeros(n),
                       length=chosen_lengths))
df.to_csv(f'{args.path}/{args.set}.csv')
pd.DataFrame(np.zeros((1, 300))).to_csv(f'{args.path}/vectors.csv')
