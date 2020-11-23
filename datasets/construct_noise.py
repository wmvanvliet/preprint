"""
Construct a dataset containing 128x128 pixel images of random pixels.
"""
# encoding: utf-8
import argparse
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import pandas as pd
from os import makedirs
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import tfrecord

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate the noise dataset')
parser.add_argument('path', type=str, help='The path to write the dataset to.')
parser.add_argument('set', type=str, help='Specify either "train" to generate the training set and to "test" to generate the test set.')
args = parser.parse_args()

data_path = '/m/nbe/scratch/reading_models'

noise_levels = [0.1, 0.2, 0.3]

rng = np.random.RandomState(0)

# Render the consonant strings as PNG files
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

    noise_level = rng.choice(noise_levels)

    ax = f.add_axes([0, 0, 1, 1], label='noise')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    noise = rng.rand(256, 256)
    ax.imshow(noise, extent=[0, 1, 0, 1], cmap='gray', alpha=noise_level)

    canvas.draw()
    buffer, (width, height) = canvas.print_to_buffer()
    image = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))[:, :, :3]
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
df = pd.DataFrame(dict(noise=chosen_noise_levels, label=np.zeros(n)))
df.to_csv(f'{args.path}/{args.set}.csv')
pd.DataFrame(np.zeros((1, 300))).to_csv(f'{args.path}/vectors.csv')
