"""
Construct a dataset containing 128x128 pixel images of random symbol strings.  Uses the
11 symbols used in the "epasana" study.
"""
# encoding: utf-8
import argparse
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import pandas as pd
from os import makedirs
from tqdm import tqdm
import pickle
from PIL import Image
from io import BytesIO
import random
import tfrecord

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate epasana-symbols dataset')
parser.add_argument('path', type=str, help='The path to write the dataset to.')
parser.add_argument('set', type=str, help='Specify either "train" to generate the training set and to "test" to generate the test set.')
args = parser.parse_args()

# Limits
lengths = [7,8]
rotations = np.linspace(-20, 20, 11)

# Loading array of symbol bitmaps from pickle object. The pickle object is generated with symbol_bitmaps.py
with open('/m/nbe/scratch/reading_models/datasets/symbol-bitmaps', 'rb') as f:
    symbol_bitmaps = pickle.load(f)
          
rng = np.random.RandomState(0)

chosen_rotations = []
chosen_lengths = []


n = 50_000 if args.set == 'train' else 5_000

labels = np.zeros(n, dtype=np.int)

makedirs(args.path, exist_ok=True)
writer = tfrecord.TFRecordWriter(f'{args.path}/{args.set}.tfrecord')

dpi = 96.
f = Figure(figsize=(256 / dpi, 256 / dpi), dpi=dpi)
canvas = FigureCanvasAgg(f)


for i in tqdm(range(n), total=n):

    f.clf()
    ax = f.add_axes([0, 0, 1, 1])
    rotation = rng.choice(rotations)
    length = rng.choice(lengths)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Creating symbol image as numpy array
    chosen_symbols = []
    for a in range(length):
        chosen_symbols.append(random.choice(symbol_bitmaps))

    img_array = np.concatenate(chosen_symbols,axis=1)

    # Separating black pixels in symbols from black pixels that append when rotating symbol image
    img_array = img_array + 1

    # Creating image object from the numpy array
    img = Image.fromarray(img_array.astype(np.uint8))
    
    # Rotating the image
    img = img.rotate(rotation, expand='True')

    # Converting image back to numpy array
    rotated = np.asarray(img, dtype='float64').copy()

    # Changing black background pixels to white
    rotated[rotated == 0] = 106

    # Returning other pixels to their original values
    rotated = rotated - 1

    # Making a sublot within the excisting figure
    ax1 = f.add_subplot(111)
    ax1.axis('off')
    ax1.imshow(rotated,cmap = 'gray')

    canvas.draw()
    buffer, (width, height) = canvas.print_to_buffer()
    image = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))[:, :, :3]

    chosen_rotations.append(rotation)
    chosen_lengths.append(length)

    buf = BytesIO()
    Image.fromarray(image.astype(np.uint8)).save(buf, format='jpeg')

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
df = pd.DataFrame(dict(text=['symbol'] * n, rotation=chosen_rotations, label=np.zeros(n), length=chosen_lengths))
df.to_csv(f'{args.path}/{args.set}.csv')
pd.DataFrame(np.zeros((1, 300))).to_csv(f'{args.path}/vectors.csv')
