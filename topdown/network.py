import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import string
from itertools import combinations_with_replacement

from letter import letters, plot_letter

words = list(combinations_with_replacement(string.ascii_lowercase, 4))

def activation(x):
    #return x
    return np.maximum(x, 0)
    #x = x / np.abs(x).max()
    #return 1 / (1 + np.exp(-x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_word(word):
    return np.hstack([letters[l] for l in word]).astype(float)

def plot_vis_layer(vis_layer):
    fig, axes = plt.subplots(1, 4)
    for ax in axes:
        ax.set_axis_off()
    plot_letter(vis_layer[0*16:1*16], axes[0], vmax=vis_layer.max())
    plot_letter(vis_layer[1*16:2*16], axes[1], vmax=vis_layer.max())
    plot_letter(vis_layer[2*16:3*16], axes[2], vmax=vis_layer.max())
    plot_letter(vis_layer[3*16:4*16], axes[3], vmax=vis_layer.max())

def plot_letter_layer(letter_layer):
    fig, axes = plt.subplots(4, 26)
    for i, l in enumerate(letters.values()):
        axes[0, i].set_axis_off()
        axes[1, i].set_axis_off()
        axes[2, i].set_axis_off()
        axes[3, i].set_axis_off()
        plot_letter(np.array(l) * letter_layer[i + 0 * 26], axes[0, i], vmax=letter_layer.max())
        plot_letter(np.array(l) * letter_layer[i + 1 * 26], axes[1, i], vmax=letter_layer.max())
        plot_letter(np.array(l) * letter_layer[i + 2 * 26], axes[2, i], vmax=letter_layer.max())
        plot_letter(np.array(l) * letter_layer[i + 3 * 26], axes[3, i], vmax=letter_layer.max())

single_letter = np.vstack(list(letters.values()))
single_letter_error = 1 - single_letter
#single_letter[single_letter == 0] = -1
#single_letter = single_letter / np.abs(single_letter).sum(axis=0, keepdims=True)
vis2letter = block_diag(single_letter, single_letter, single_letter, single_letter).T.astype(float)
vis2letter_error = block_diag(single_letter_error, single_letter_error, single_letter_error, single_letter_error).T.astype(float)
letter2word = []
for word in words:
    ws = []
    for l in word:
        w = np.zeros(26, dtype=float)
        w[string.ascii_lowercase.index(l)] = 1
        ws.append(w)
    letter2word.append(np.hstack(ws)[None, :])
letter2word = np.vstack(letter2word).T
#letter2word /= np.abs(letter2word).sum(axis=0, keepdims=True)

##
clamp = load_word('word')
vis_layer = clamp
vis_error_signal = np.zeros_like(clamp)

for steps in range(1):
    # First forward pass
    vis_layer = clamp
    letter_layer = activation(vis_layer @ vis2letter)
    word_layer = activation(letter_layer @ letter2word)

    # Backward pass, letter layer only
    top_down_vis_layer = activation(letter_layer[:, None] * vis2letter.T)
    vis_error_signal = activation((1 - vis_layer) - top_down_vis_layer)
    letter_error_signal = vis_error_signal @ vis2letter
    print(letter_error_signal)
