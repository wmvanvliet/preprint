"""
Feed the images used in the MEG experiment through the model and export the
layer activations in a format suitable for loading into TensorFlow's Embedding
Projector tool.
"""
import torch
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
import json
import mkl
mkl.set_num_threads(4)

import networks
from pilot import utils

# The model to perform the analysis on. I keep changing this around as I train new models.
model_name = 'vgg_first_imagenet64_then_tiny-words-noisy_tiny-imagenet'

# Get the images that were presented during the MEG experiment
stimuli = utils.get_stimulus_info(subject=2)
images = utils.get_stimulus_images(subject=2, stimuli=stimuli)

# Load the model and feed through the images
checkpoint = torch.load('/m/nbe/scratch/reading_models/models/%s.pth.tar' % model_name, map_location='cpu')
model = networks.vgg.from_checkpoint(checkpoint)
feature_outputs, classifier_outputs = model.get_layer_activations(images)
outputs = feature_outputs + classifier_outputs
names = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2', 'output']

# Save layer outputs to projector
embeddings = []
for name, out in zip(names, outputs):
    out = out.reshape(360, -1)
    print(f'/m/nbe/scratch/reading_models/projector/{model_name}_{name}.tsv')
    np.savetxt(f'/m/nbe/scratch/reading_models/projector/{model_name}_{name}.tsv', out, delimiter='\t')

    embeddings.append(dict(
        tensorName=name,
        tensorShape=list(out.shape),
        tensorPath=f'{model_name}_{name}.tsv',
        metadataPath='metadata.tsv',
        sprite=dict(
            imagePath='thumbnails.png',
            singleImageDim=[64, 64],
        )
    ))

# Save metadata
stimuli.to_csv('/m/nbe/scratch/reading_models/projector/metadata.tsv', sep='\t')

# Save images to projector
img = Image.fromarray((make_grid(images/5 + 0.5, nrow=20, padding=0).numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
img.save('/m/nbe/scratch/reading_models/projector/thumbnails.png', format='png')

# Compile JSON describing the entire dataset
with open(f'/m/nbe/scratch/reading_models/projector/{model_name}.json', 'w') as f:
    json.dump(dict(embeddings=embeddings[::-1]), f, indent=False)
