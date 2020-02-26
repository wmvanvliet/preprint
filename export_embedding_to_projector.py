import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
import mne
import numpy as np
import os
import json
import mkl
mkl.set_num_threads(4)

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['VECLIB_MAXIMUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import networks

#model_name = 'vgg_first_imagenet64_then_tiny-words_tiny-consonants_tiny-imagenet'
model_name = 'vgg_first_imagenet64_then_tiny-words_w2v'

epochs = mne.read_epochs('data/pilot_data/pilot2/pilot2_epo.fif', preload=False)
epochs = epochs[['word', 'symbols', 'consonants']]
stimuli = epochs.metadata.groupby('text').agg('first').sort_values('event_id')
stimuli['y'] = np.arange(len(stimuli))

preproc = transforms.Compose([
    transforms.CenterCrop(208),
    transforms.Resize(64),
    transforms.CenterCrop(60),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

unnormalize = transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                   std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

filesizes = []
images = []
for fname in tqdm(stimuli['filename'], desc='Reading images'):
    with Image.open(f'data/pilot_data/pilot2/stimuli/{fname}') as image:
        image = image.convert('RGB')
        image = preproc(image).unsqueeze(0)
        #image = transforms.ToTensor()(image)
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        images.append(image)
images = torch.cat(images, 0)

checkpoint = torch.load('data/models/%s.pth.tar' % model_name, map_location='cpu')
num_classes = checkpoint['state_dict']['classifier.6.weight'].shape[0]
classifier_size = checkpoint['state_dict']['classifier.6.weight'].shape[1]
model = networks.vgg(num_classes=num_classes, classifier_size=classifier_size)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

outputs = []
out = images
for i, layer in enumerate(model.features):
    layer_out = []
    for j in range(0, len(out), 128):
        layer_out.append(layer(out[j:j + 128]))
    out = torch.cat(layer_out, 0)
    del layer_out
    print('layer %02d, output=%s' % (i, out.shape))
    if i in [4, 10, 17, 24]:
        outputs.append(out.detach().numpy().copy())
out = out.view(out.size(0), -1)
for i, layer in enumerate(model.classifier):
    layer_out = []
    for j in range(0, len(out), 128):
        layer_out.append(layer(out[j:j + 128]))
    out = torch.cat(layer_out, 0)
    print('layer %02d, output=%s' % (i, out.shape))
    if i in [1, 4, 6]:
        outputs.append(out.detach().numpy().copy())

names = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2', 'output']

# Save layer outputs to projector
embeddings = []
for name, out in zip(names, outputs):
    out = out.reshape(360, -1)
    print(f'projector/{model_name}_{name}.tsv')
    np.savetxt(f'projector/{model_name}_{name}.tsv', out, delimiter='\t')

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
stimuli.to_csv('projector/metadata.tsv', sep='\t')

# Save images to projector
img = Image.fromarray((make_grid(images/5 + 0.5, nrow=20, padding=0).numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
img.save('projector/thumbnails.png', format='png')

# Compile JSON describing the entire dataset
with open(f'projector/{model_name}.json', 'w') as f:
    json.dump(dict(embeddings=embeddings[::-1]), f, indent=False)
