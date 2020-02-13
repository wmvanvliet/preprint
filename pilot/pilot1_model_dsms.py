import torch
import torchvision.transforms as transforms
import rsa
import mne
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import networks
import pickle
from matplotlib import pyplot as plt
from scipy.spatial import distance

epochs = mne.read_epochs('../data/pilot_data/pilot1/pilot1_epo.fif', preload=False)
epochs = epochs[['word', 'symbols', 'consonants']]
stimuli = epochs.metadata.groupby('text').agg('first').sort_values('event_id')
stimuli['y'] = np.arange(len(stimuli))
metadata = pd.merge(epochs.metadata, stimuli[['y']], left_on='text', right_index=True).sort_index()
assert np.array_equal(metadata.event_id.values.astype(int), epochs.events[:, 2])

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

images = []
for fname in tqdm(stimuli['filename'], desc='Reading images'):
    with Image.open(f'../data/pilot_data/pilot1/stimuli/{fname}') as image:
        image = image.convert('RGB')
        image = preproc(image).unsqueeze(0)
        #image = transforms.ToTensor()(image)
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        images.append(image)
images = torch.cat(images, 0)

model_name = 'vgg_first_imagenet64_then_tiny-words_tiny-imagenet'
checkpoint = torch.load(f'../data/models/{model_name}.pth.tar', map_location=torch.device('cpu'))
num_classes = checkpoint['state_dict']['classifier.6.weight'].shape[0]
model = networks.vgg(num_classes=num_classes)
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

def words_only(x, y):
    if x != 'word' or y != 'word':
        return 1
    else:
        return 0

def letters_only(x, y):
    if x == 'symbols' or y == 'symbols':
        return 1
    else:
        return 0

print('Computing model DSMs...', end='', flush=True)
dsm_models = [
    rsa.compute_dsm(feature_outputs[0], metric='correlation'),
    rsa.compute_dsm(feature_outputs[1], metric='correlation'),
    rsa.compute_dsm(feature_outputs[2], metric='correlation'),
    rsa.compute_dsm(feature_outputs[3], metric='correlation'),
    rsa.compute_dsm(classifier_outputs[0], metric='correlation'),
    rsa.compute_dsm(classifier_outputs[1], metric='correlation'),
    rsa.compute_dsm(classifier_outputs[2], metric='correlation'),
    rsa.compute_dsm(stimuli[['type']], metric=words_only),
    rsa.compute_dsm(stimuli[['type']], metric=letters_only),
    rsa.compute_dsm(images.numpy().reshape(len(images), -1), metric='euclidean'),
]
dsm_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Classifier 1', 'Classifier 2', 'Classifier 3',
             'Words only', 'Letters only', 'Pixels']

with open(f'../data/dsms/pilot1_{model_name}_dsms.pkl', 'wb') as f:
    pickle.dump(dict(dsms=dsm_models, dsm_names=dsm_names), f)

fig, ax = plt.subplots(3, 4, sharex='col', sharey='row', figsize=(10, 10))
for row in range(3):
    for col in range(4):
        i = row * 4 + col
        if i < len(dsm_models):
            ax[row, col].imshow(distance.squareform(dsm_models[i]), cmap='magma')
            ax[row, col].set_title(dsm_names[i])
plt.tight_layout()
plt.savefig(f'../figures/pilot1_dsms_{model_name}.pdf')
