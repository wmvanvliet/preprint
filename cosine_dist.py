import torch
import torchvision.transforms as transforms
import mne
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.io import loadmat
from matplotlib import pyplot as plt
from scipy.spatial import distance

import networks

epochs = mne.read_epochs('data/pilot_data/pilot2/pilot2_epo.fif', preload=False)
epochs = epochs[['word', 'symbols', 'consonants']]
stimuli = epochs.metadata.groupby('text').agg('first').sort_values('event_id')
stimuli['y'] = np.arange(len(stimuli))
metadata = pd.merge(epochs.metadata, stimuli[['y']], left_on='text', right_index=True).sort_index()
assert np.array_equal(metadata.event_id.values.astype(int), epochs.events[:, 2])

# Get word2vec vectors for the words
word2vec = loadmat('data/word2vec.mat')
vocab = [v.strip() for v in word2vec['vocab']]
word_vectors = []
non_word_vectors = []
for text, [type] in stimuli[['type']].iterrows():
    if type == 'word':
        word_vectors.append(word2vec['vectors'][vocab.index(text)])
    else:
        vector = np.zeros(300, dtype=word2vec['vectors'].dtype)
        non_word_vectors.append(vector)
word_vectors = np.array(word_vectors)
non_word_vectors = np.array(non_word_vectors)
vectors = np.vstack((word_vectors, non_word_vectors))

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
    with Image.open(f'data/pilot_data/pilot2/stimuli/{fname}') as image:
        image = image.convert('RGB')
        image = preproc(image).unsqueeze(0)
        #image = transforms.ToTensor()(image)
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        images.append(image)
images = torch.cat(images, 0)

model_name = 'vgg_first_imagenet64_then_tiny-words-noisy_tiny-imagenet'
#model_name = 'vgg_first_imagenet64_then_tiny-words_tiny-imagenet'
checkpoint = torch.load(f'data/models/{model_name}.pth.tar', map_location=torch.device('cpu'))
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

noise = np.random.randn(180, 300)
noise2 = np.random.randn(360, 400)

#for noise_lvl in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0]:
noise_lvl = 0.1

desired_word_vectors = word_vectors + 0 * noise
desired_non_word_vectors = non_word_vectors + 0.1 * noise
model_vectors = classifier_outputs[-1] + noise_lvl * noise2
model_word_vectors = model_vectors[:180]
model_non_word_vectors = model_vectors[180:]

d1 = distance.pdist(desired_word_vectors, metric='cosine')
d2 = distance.pdist(desired_non_word_vectors, metric='cosine')
d3 = distance.pdist(model_word_vectors, metric='cosine')
d4 = distance.pdist(model_non_word_vectors, metric='cosine')

print(noise_lvl, d3.mean() - d4.mean())

bins = np.arange(0, 2, 0.02)

f = plt.figure(figsize=(12, 10))
ax = f.subplots(4, 1, sharex=True, sharey=True)
ax[0].hist(d1, bins, color='C0')
ax[0].axvline(1, color='black')
ax[0].axvline(d1.mean(), color='C0')
ax[0].set_title('True word2vec vectors for words')
ax[1].hist(d2, bins, color='C1')
ax[1].axvline(d2.mean(), color='C1')
ax[1].axvline(1, color='black')
ax[1].set_title('Simulated, random, vectors for non-words')
ax[2].hist(d3, bins, color='C2')
ax[2].axvline(d3.mean(), color='C2')
ax[2].axvline(1, color='black')
ax[2].set_title('Model outputs on words')
ax[3].hist(d4, bins, color='C3')
ax[3].axvline(d4.mean(), color='C3')
ax[3].axvline(1, color='black')
ax[3].set_title('Model outputs on non-words')
plt.xlabel('Cosine distance')
plt.xlim(0, 2)
plt.tight_layout()

f = plt.figure(figsize=(6, 10))
ax = f.subplots(2, 1, sharex=True, sharey=True)
im1 = ax[0].imshow(np.vstack((desired_word_vectors, desired_non_word_vectors)), cmap='RdBu_r', vmin=-1, vmax=1)
ax[0].axhline(180, color='black')
ax[0].axhline(180 + 90, color='black')
ax[0].set_title('Desired output')
f.colorbar(im1, ax=ax[0])
im2 = ax[1].imshow(np.vstack((model_word_vectors, model_non_word_vectors)), cmap='RdBu_r', vmin=-1, vmax=1)
ax[1].axhline(180, color='black')
ax[1].axhline(180 + 90, color='black')
ax[1].set_title('Actual model output')
f.colorbar(im2, ax=ax[1])
plt.tight_layout()

plt.figure()
plt.imshow(distance.squareform(d4))
