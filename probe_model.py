import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
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

model_name = 'quickdraw_image_redness1'

preproc = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(60),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225]),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.2, 0.2, 0.2]),
])

dpi = 96.
f = Figure(figsize=(64 / dpi, 64 / dpi), dpi=dpi)
canvas = FigureCanvasAgg(f)

def make_image(word='koira', rotation=0, size=16, family='arial', fname=None, noise=0):
    f.clf()
    ax = f.add_axes([0, 0, 1, 1])
    fontprop = fm.FontProperties(family=family, fname=fname)
    background = plt.Rectangle((0, 0), 64, 64, facecolor=(0.5, 0.5, 0.5, 1.0), zorder=0)
    ax.add_patch(background)
    noise_image = np.random.randn(64, 64)
    ax.imshow(noise_image, extent=[0, 1, 0, 1], cmap='gray', alpha=noise, zorder=1)
    ax.text(0.5, 0.5, word, ha='center', va='center',
            rotation=rotation, fontsize=size, fontproperties=fontprop, alpha=1 - noise, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    canvas.draw()
    buffer, (width, height) = canvas.print_to_buffer()
    image = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))[:, :, :3]
    #image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())

    image = Image.fromarray(image)
    #image = 2.64 - preproc(image).unsqueeze(0)
    image = preproc(image).unsqueeze(0)
    return image

# Construct images
images = []
for noise in np.arange(0, 1.0, 0.01):
    images.append(make_image(word='koira', noise=noise))
images = torch.cat(images, 0)

plt.figure()
plt.imshow(make_grid(images/5 + 0.5, nrow=10).numpy().transpose(1, 2, 0))

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

plt.figure()
plt.subplot(221)
plt.plot(np.abs(feature_outputs[0]).sum(axis=(1, 2, 3)), color='C0')
plt.subplot(222)
plt.plot(np.abs(feature_outputs[1]).sum(axis=(1, 2, 3)), color='C1')
plt.subplot(223)
plt.plot(np.abs(feature_outputs[2]).sum(axis=(1, 2, 3)), color='C2')
plt.subplot(224)
plt.plot(np.abs(feature_outputs[3]).sum(axis=(1, 2, 3)), color='C3')
