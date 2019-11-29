import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from scipy.io import loadmat
import rsa
import mne
import numpy as np
import sys
import os
from matplotlib import pyplot as plt
import mkl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib import font_manager as fm
mkl.set_num_threads(4)

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import networks

model_name = 'vgg_quickdraw_image_redness1'

stimuli = pd.read_csv('/l/vanvlm1/redness1/stimuli.csv', index_col=0)

# Construct images
images = []

dpi = 96.
f = Figure(figsize=(64 / dpi, 64 / dpi), dpi=dpi)
canvas = FigureCanvasAgg(f)
for noise_level in np.arange(0, 1, 0.1):
    f.clf()
    ax = f.add_axes([0, 0, 1, 1])
    word = 'koira'
    rotation = 0
    fontsize = 10
    fontfamily, fontfile = 'arial', None
    fontprop = fm.FontProperties(family=fontfamily, fname=fontfile)
    noise_level = noise_level
    noise = np.random.rand(64, 64)
    ax.imshow(noise, extent=[0, 1, 0, 1], cmap='gray', alpha=noise_level)
    ax.text(0.5, 0.5, word, ha='center', va='center',
            rotation=rotation, fontsize=fontsize, fontproperties=fontprop, alpha=1 - noise_level)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    image = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    images.append(image)
images = np.array(images)

preproc = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(60),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

checkpoint = torch.load('models/%s.pth.tar' % model_name, map_location='cpu')
num_classes = checkpoint['state_dict']['classifier.6.weight'].shape[0]
#num_classes = checkpoint['state_dict']['classifier.weight'].shape[0]
model = networks.vgg(num_classes=num_classes)
#model = networks.TwoLayerNet(num_classes=num_classes)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

feature_outputs = []
out = images
for i, layer in enumerate(model.features):
    layer_out = []
    for j in range(0, len(out), 128):
        layer_out.append(layer(out[j:j + 128]))
    out = torch.cat(layer_out, 0)
    del layer_out
    print('layer %02d, output=%s' % (i, out.shape))
    if i in [4, 10, 17, 24]:
        feature_outputs.append(out.detach().numpy().copy())
classifier_outputs = []
out = out.view(out.size(0), -1)
layer_out = []
for i, layer in enumerate(model.classifier):
    layer_out = []
    for j in range(0, len(out), 128):
        layer_out.append(layer(out[j:j + 128]))
    out = torch.cat(layer_out, 0)
    print('layer %02d, output=%s' % (i, out.shape))
    if i in [1, 4, 6]:
        classifier_outputs.append(out.detach().numpy().copy())
#classifier_outputs.append(model.classifier(out).detach().numpy().copy())
