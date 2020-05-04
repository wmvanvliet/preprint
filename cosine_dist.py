"""
Experimentations in trying to get the final (=semantic) layer of the model to
match the N400 potential.
"""
import torch
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from scipy.spatial import distance

import networks
from pilot import utils

stimuli = utils.get_stimulus_info(subject=2, data_path='M:/scratch/reading_models')
images = utils.get_stimulus_images(subject=2, stimuli=stimuli, data_path='M:/scratch/reading_models')

# Get word2vec vectors for the words
word2vec = loadmat('M:/scratch/reading_models/word2vec.mat')
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

#model_name = 'vgg_first_imagenet64_then_tiny-words-noisy_tiny-imagenet'
model_name = 'vgg_first_imagenet64_then_tiny-words_tiny-consonants_tiny-symbols_tiny-imagenet_w2v'
checkpoint = torch.load('M:/scratch/reading_models/models/%s.pth.tar' % model_name, map_location='cpu')
model = networks.vgg_sem.from_checkpoint(checkpoint, freeze=True)
classifier_outputs = model.get_layer_activations(images)[-1]
model_output = classifier_outputs[-1]

noise = np.random.randn(180, 300)
noise2 = np.random.randn(360, 300)

for noise_lvl in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    #noise_lvl = 0.1

    desired_word_vectors = word_vectors + 0 * noise
    desired_non_word_vectors = non_word_vectors + 0.1 * noise

    model_vectors = classifier_outputs[-1] + noise_lvl * noise2
    model_word_vectors = model_vectors[:180]
    model_non_word_vectors = model_vectors[180:]

    d1 = distance.pdist(desired_word_vectors, metric='correlation')
    d2 = distance.pdist(desired_non_word_vectors, metric='correlation')
    d3 = distance.pdist(model_word_vectors, metric='correlation')
    d4 = distance.pdist(model_non_word_vectors, metric='correlation')
    d5 = distance.pdist(model_vectors, metric='correlation')

    print(noise_lvl, d3.mean() - d4.mean())
noise_lvl = 0.1

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
im1 = ax[0].imshow(np.vstack((desired_word_vectors, desired_non_word_vectors)))
ax[0].axhline(180, color='black')
ax[0].axhline(180 + 90, color='black')
ax[0].set_title('Desired output')
f.colorbar(im1, ax=ax[0])
im2 = ax[1].imshow(np.vstack((model_word_vectors, model_non_word_vectors)))
ax[1].axhline(180, color='black')
ax[1].axhline(180 + 90, color='black')
ax[1].set_title('Actual model output')
f.colorbar(im2, ax=ax[1])
plt.tight_layout()

plt.figure()
plt.imshow(distance.squareform(d5))
