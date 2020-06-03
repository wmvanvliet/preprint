"""
Construct a dataset containing 128x128 pixel images of random consonant strings.  Uses the
consonants used in the epasana study.
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
parser = argparse.ArgumentParser(description='Generate the consonants dataset')
parser.add_argument('path', type=str, help='The path to write the dataset to.')
parser.add_argument('set', type=str, help='Specify either "train" to generate the training set and to "test" to generate the test set.')
args = parser.parse_args()

data_path = '/m/nbe/scratch/reading_models'

consonants = list('BDFGHJKLMNPRSTV')

rotations = np.linspace(-20, 20, 11)
sizes = np.linspace(7, 16, 21)
fonts = ['courier new', 'dejavu sans mono', 'times new roman', 'arial',
         'arial black', 'verdana', 'comic sans ms', 'georgia',
         'liberation serif', 'impact', 'roboto condensed']
noise_levels = [0.2, 0.35, 0.5]
lengths = [7, 8]

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
chosen_noise_levels = []
chosen_lengths = []

n = 100_000 if args.set == 'train' else 10_000
data = []
labels = np.zeros(n, dtype=np.int)

dpi = 96.
f = Figure(figsize=(128 / dpi, 128 / dpi), dpi=dpi)
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
    noise = rng.rand(128, 128)
    ax.imshow(noise, extent=[0, 1, 0, 1], cmap='gray', alpha=noise_level)

    # Generate the consonant string
    text = ''.join(rng.choice(consonants, length))
    ax.text(0.5, 0.5, text, ha='center', va='center',
            rotation=rotation, fontproperties=fontprop, alpha=1 - noise_level)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    #f.savefig('training_imgs/imgs' + '/'+text+'%03d.JPEG' % i)

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
    Image.fromarray(image.astype(np.uint8)).save(buf, format='png')
    img_compressed = buf.getvalue()

    data.append(img_compressed)

df = pd.DataFrame(dict(text=chosen_strings, rotation=chosen_rotations,
                       size=chosen_sizes, font=chosen_fonts, label=labels,
                       noise_level=chosen_noise_levels, lenght=chosen_lengths))

makedirs(args.path, exist_ok=True)

df.to_csv(f'{args.path}/{args.set}.csv')
with open(f'{args.path}/{args.set}', 'wb') as f:
    pickle.dump(dict(data=data, labels=labels), f)

#with open(f'{args.path}/meta', 'wb') as f:
#    pickle.dump(dict(label_names=['consonants'], vectors=np.zeros((1, 300))), f)
