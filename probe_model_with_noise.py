"""
Feed images with varying levels of noise through the model.
"""
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib import font_manager as fm

import networks

model_name = 'vgg_first_imagenet64_then_tiny-words-noisy_tiny-imagenet'

# The preprocessing transform used during training of the model
preproc = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(60),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.2, 0.2, 0.2]),
])

# Code to generate images starts here.
#
# Matplotlib is used to generate them, which is fast enough for our purposes if
# we take care to re-use the same figure.

# Create figure of exactly 64x64 pixels
dpi = 96.
f = Figure(figsize=(64 / dpi, 64 / dpi), dpi=dpi)
# We need the canvas object to get the bitmap data at the end
canvas = FigureCanvasAgg(f)  

# Use a single noise image. Alternatively, consider generating a different
# noise image every iteration.
noise_image = np.random.randn(64, 64)
def make_image(word, rotation=0, size=16, family='dejavu sans', fname=None, noise=0):
    # Initialize an empty figure
    f.clf()
    ax = f.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Start with a grey background image
    background = plt.Rectangle((0, 0), 64, 64, facecolor=(0.5, 0.5, 0.5, 1.0), zorder=0)
    ax.add_patch(background)

    # Add noise to the image. Note the usage of the alpha parameter to tweak the amount of noise
    ax.imshow(noise_image, extent=[0, 1, 0, 1], cmap='gray', alpha=noise, zorder=1)
    
    # Add the text to the image in the selected font
    fontprop = fm.FontProperties(family=family, fname=fname, size=size)
    ax.text(0.5, 0.5, word, ha='center', va='center',
            rotation=rotation, fontproperties=fontprop, alpha=1 - noise, zorder=2)

    # Render the image and create a PIL.Image from the pixel data
    canvas.draw()
    buffer, (width, height) = canvas.print_to_buffer()
    image = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))[:, :, :3]
    image = Image.fromarray(image)

    # Transform the PIL image to a PyTorch tensor for feeding into the model
    image = preproc(image).unsqueeze(0)
    return image

# Construct images
images = []
#for noise in np.arange(0, 1.0, 0.01):
for size in np.linspace(5, 16, 100):
    images.append(make_image(word='hello', noise=0, size=size))
images = torch.cat(images, 0)

# Plot the images
plt.figure(figsize=(10, 10))
plt.imshow(make_grid(images/5 + 0.5, nrow=10).numpy().transpose(1, 2, 0))
plt.axis('off')
plt.tight_layout()

# Load the model and feed through the images
model_name = 'vgg_first_imagenet64_then_tiny-words-noisy_tiny-imagenet'
checkpoint = torch.load('/m/nbe/scratch/reading_models/models/%s.pth.tar' % model_name, map_location='cpu')
model = networks.vgg.from_checkpoint(checkpoint)
feature_outputs, classifier_outputs = model.get_layer_activations(images)

# Plot the activation of the layers for varying amounts of noise
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
