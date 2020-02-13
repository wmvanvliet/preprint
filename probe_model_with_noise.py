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

model_name = 'vgg_first_imagenet64_then_tiny-words_tiny-imagenet'
#model_name = 'vgg_tiny_redness1_image'

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

noise_image = np.random.randn(64, 64)
def make_image(word='koira', rotation=0, size=16, family='dejavu sans', fname=None, noise=0):
    f.clf()
    ax = f.add_axes([0, 0, 1, 1])
    fontprop = fm.FontProperties(family=family, fname=fname)
    background = plt.Rectangle((0, 0), 64, 64, facecolor=(0.5, 0.5, 0.5, 1.0), zorder=0)
    ax.add_patch(background)
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

plt.figure(figsize=(10, 10))
plt.imshow(make_grid(images/5 + 0.5, nrow=10).numpy().transpose(1, 2, 0))
plt.axis('off')
plt.tight_layout()

checkpoint = torch.load('data/models/%s.pth.tar' % model_name, map_location='cpu')
num_classes = checkpoint['state_dict']['classifier.6.weight'].shape[0]
classifier_size = checkpoint['state_dict']['classifier.6.weight'].shape[1]
#num_classes = checkpoint['state_dict']['classifier.weight'].shape[0]
model = networks.vgg(num_classes=num_classes, classifier_size=classifier_size)
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

plt.figure(figsize=(12, 5))
plt.subplot(241)
plt.plot(np.abs(feature_outputs[0]).mean(axis=(1, 2, 3)), color='C0')
plt.title('Feature 1')
plt.subplot(242)
plt.plot(np.abs(feature_outputs[1]).mean(axis=(1, 2, 3)), color='C1')
plt.title('Feature 2')
plt.subplot(243)
plt.plot(np.abs(feature_outputs[2]).mean(axis=(1, 2, 3)), color='C2')
plt.title('Feature 3')
plt.subplot(244)
plt.plot(np.abs(feature_outputs[3]).mean(axis=(1, 2, 3)), color='C3')
plt.title('Feature 4')
plt.subplot(245)
plt.plot(np.abs(classifier_outputs[0]).mean(axis=1), color='C4')
plt.title('Classifier 1')
plt.subplot(246)
plt.plot(np.abs(classifier_outputs[1]).mean(axis=1), color='C5')
plt.title('Classifier 2')
plt.subplot(247)
plt.plot(np.abs(classifier_outputs[2]).mean(axis=1), color='C6')
plt.title('Classifier 3')
plt.tight_layout()
