"""
3 blocks of 15 minutes each. 15 * 60 = 600 seconds
180 Word trials of 1.5 seconds: 270 seconds
90 Consonant string trials of 1.5 seconds: 135 seconds
90 Symbol string trials of 1.5 seconds: 135 seconds
60 Question trials of 1 second: 60 seconds

1080 trials in total
540 words
270 consonant strings
270 symbol strings

180 question trials

stimulus length: 3-5 letters

question: is the given symbol in the correct location?
stimulus: KOIRA
question: _ O _ _ _ (correct)
          _ _ _ M _ (incorrect)
"""
#encoding: utf8
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import itertools
from scipy.io import loadmat
from scipy.spatial import distance
from libvoikko import Voikko

# Use a very specific random generator with a specific seed so in the future,
# this script will still generate the same stimuli.
rng = np.random.Generator(np.random.PCG64(18420))

# Setup Finnish and English dictionaries
dict_fi = pd.read_csv('data/fi_full.txt', sep=' ', names=['word', 'freq'], dtype='str', keep_default_na=False, index_col=0, nrows=500_000)
dict_en = pd.read_csv('data/en_full.txt', sep=' ', names=['word', 'freq'], dtype='str', keep_default_na=False, index_col=0)
Voikko.setLibrarySearchPath('data/voikko')
spell_dict_fi = Voikko('fi_FI', 'data/voikko')

# Setup word2vec
m = loadmat('data/word2vec.mat')
w2v_vocab = {word.strip(): i for i, word in enumerate(m['vocab'])}

alphabet = list('abcdefghijklmnopqrstuvwxyzäöü')
consonants = list('bcdfghjklmnpqrstvwxz')

# Filter Finnish words
sel = np.array([np.all([
    4 <= len(word) <= 6,
    np.all([letter in alphabet for letter in word]),
    word not in dict_en.index,
    word in w2v_vocab,
    spell_dict_fi.spell(word),
]) for word in tqdm(dict_fi.index)])

# Filter the word2vec data based on the selection defined above
vocab = [word for i, word in enumerate(dict_fi.index) if sel[i]]
freqs = dict_fi['freq'][sel].values.astype(int)
vectors = m['vectors'][[w2v_vocab[w] for w in vocab]]

# Create some selectors for different word lengths
vocab_sel_4 = np.flatnonzero([len(word) == 4 for word in vocab])
vocab_sel_5 = np.flatnonzero([len(word) == 5 for word in vocab])
vocab_sel_6 = np.flatnonzero([len(word) == 6 for word in vocab])

symbols = {
    's': '\u25FB', # Square
    'o': '\u25CB', # Circle
    '^': '\u25B3', # Triangle up
    'v': '\u25BD', # Triangle down
    'd': '\u25C7', # Diamond
}

rotations = [-15, 0, +15]
fontsizes = [20, 30, 40]
noise_levels = [0.2, 0.35, 0.5]
fonts = ['Comic Sans MS', 'Impact', 'Times New Roman']

def sel_words_w2v_dist(vocab_sel, n=60):
    """Select words which are spread out in w2v space"""
    sel = [vocab_sel[0]]
    while len(sel) < n:
        non_sel = np.setdiff1d(vocab_sel, sel)
        # Distance to all the selected words in w2v space
        vectors_sel = vectors[sel]
        vectors_non_sel = vectors[non_sel]
        next_sel = distance.cdist(vectors_sel, vectors_non_sel).sum(axis=0).argmax()
        sel.append(non_sel[next_sel])
    return np.array(sel)

n_word_strings = 180
word_sel = np.hstack([sel_words_w2v_dist(vocab_sel_4, 60),
                      sel_words_w2v_dist(vocab_sel_5, 60),
                      sel_words_w2v_dist(vocab_sel_6, 60)])
rng.shuffle(word_sel)
word_strings = [vocab[i] for i in word_sel]
word_freqs = freqs[word_sel]
word_vectors = vectors[word_sel]

n_symbol_strings = 90
symbol_strings = list(set(''.join(rng.choice(list(symbols.keys()), 4)) for _ in range(60)))[:30]
symbol_strings += list(set(''.join(rng.choice(list(symbols.keys()), 5)) for _ in range(60)))[:30]
symbol_strings += list(set(''.join(rng.choice(list(symbols.keys()), 6)) for _ in range(60)))[:30]
rng.shuffle(symbol_strings)

n_consonant_strings = 90
consonant_strings = list(set(''.join(rng.choice(consonants, 4)) for _ in range(60)))[:30]
consonant_strings += list(set(''.join(rng.choice(consonants, 5)) for _ in range(60)))[:30]
consonant_strings += list(set(''.join(rng.choice(consonants, 6)) for _ in range(60)))[:30]
rng.shuffle(consonant_strings)

types = (['word'] * n_word_strings) + (['symbols'] * n_symbol_strings) + (['consonants'] * n_consonant_strings)
freqs = word_freqs.tolist() + ([np.nan] * n_symbol_strings) + ([np.nan] * n_consonant_strings)
texts = word_strings + symbol_strings + consonant_strings

stimuli = []
stimuli_iter = zip(
    types, 
    texts,
    freqs,
    itertools.cycle(itertools.product(fonts, fontsizes, rotations, noise_levels)),
)
for type, text, freq, (font, fontsize, rotation, noise_level) in stimuli_iter:
    if type == 'symbols':
        font = 'DejaVu Sans'  # Symbols only render correctly in this font
    filename = f'{type}_{text}.png'
    stimuli.append(dict(type=type, text=text, freq=freq, font=font, fontsize=fontsize, rotation=rotation, noise_level=noise_level, filename=filename))
stimuli = pd.DataFrame(stimuli)

def make_question(type, text, correct):
    """Make a question to go with a stimulus."""
    pos = rng.choice(range(len(text)))
    slots = ['_'] * len(text)
    if correct:
        slots[pos] = text[pos]
    else:
        if type == 'word':
            slots[pos] = rng.choice(np.setdiff1d(alphabet, [text[pos]]))
        elif type == 'consonants':
            slots[pos] = rng.choice(np.setdiff1d(consonants, [text[pos]]))
        elif type == 'symbols':
            slots[pos] = rng.choice(np.setdiff1d(list(symbols.keys()), [text[pos]]))
        else:
            raise ValueError('Invalid stimulus type')

    return ' '.join(slots)

corrects = rng.choice([True, False], len(stimuli))
questions = [make_question(type, text, correct) for type, text, correct in zip(types, texts, corrects)]
stimuli['question'] = questions
stimuli['question_correct'] = corrects
stimuli['question_filename'] = [f'{type}_{text}_question.png' for type, text in zip(types, texts)]

event_ids = dict(word=10, consonants=20, symbols=30, question=40)
stimuli['event_id'] = [event_ids[type] for type in stimuli['type']]

assert len(np.unique(stimuli['text'])) == len(stimuli)
stimuli.to_csv('data/presentation/stimuli.csv')

# Create image files
plt.close('all')
dpi = 96.
width, height = 800, 400
f = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

# Fixation cross
plt.clf()
ax = f.add_axes([0, 0, 1, 1])
ax.set_facecolor((0.5, 0.5, 0.5))
plt.plot([0.49, 0.51], [0.5, 0.5], color='black', linewidth=1)
plt.plot([0.5, 0.5], [0.48, 0.52], color='black', linewidth=1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig('data/presentation/stimuli/fixation_cross.png')

def plot_stimulus(type, text, filename, font='DejaVu Sans', fontsize=30, rotation=0, noise_level=0):
    plt.clf()
    ax = f.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.5, 0.5, 0.5))

    if type == 'symbols':
        for k, v in symbols.items():
            text = text.replace(k, v)

    noise_level = noise_level
    noise = np.random.rand(width, height)
    ax.imshow(noise, extent=[0, 1, 0, 1], cmap='gray', alpha=noise_level, aspect='auto')

    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=fontsize,
            family=font, alpha=1 - noise_level, rotation=rotation)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(f'data/presentation/stimuli/{filename}')

def plot_question(type, question, filename, font='DejaVu Sans', fontsize=30):
    plt.clf()
    ax = f.add_axes([0, 0, 1, 1])
    ax.set_facecolor((0.5, 0.5, 0.5))

    if type == 'symbols':
        for k, v in symbols.items():
            question = question.replace(k, v)

    ax.text(0.5, 0.5, question, ha='center', va='center', family=font, fontsize=fontsize)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(f'data/presentation/stimuli/{filename}')

# Stimuli
for i, stimulus in tqdm(stimuli.iterrows(), total=len(stimuli)):
    plot_stimulus(type=stimulus['type'], text=stimulus['text'],
                  filename=stimulus['filename'], font=stimulus['font'],
                  fontsize=stimulus['fontsize'], rotation=stimulus['rotation'],
                  noise_level=stimulus['noise_level'])
    plot_question(type=stimulus['type'], question=stimulus['question'], filename=stimulus['question_filename'], font=stimulus['font'], fontsize=stimulus['fontsize'])

plt.close()

def build_run():
    run = stimuli.iloc[rng.choice(np.arange(len(stimuli)), len(stimuli), replace=False)]
    run = run.reset_index(drop=True)
    question_every = len(stimuli) // 60
    question_points = np.arange(0, len(stimuli), question_every) + rng.choice(np.arange(2, question_every), 60)
    assert np.diff(question_points).min() > 2

    run['question_asked'] = False
    run.loc[question_points, 'question_asked'] = True

    output = """
        write_codes = true;
        active_buttons = 2;
        begin;

        picture {
           default_code = "fixation";
           bitmap {
                filename = "fixation_cross.png";
           };
           x = 0; y = 0;
        } fixation;

        """
    output += 'TEMPLATE "stimulus.tem" {\n'
    output += 'word file code;\n'
    for i, stimulus in run.iterrows():
        output += f'"{stimulus["text"]}" "{stimulus["filename"]}" {stimulus["event_id"]};\n'
        if i in question_points:
            output += '};\n'
            output += '\n'
            output += 'TEMPLATE "question.tem" {\n'
            output += 'question file code;\n'
            output += f'"{stimulus["question"]}" "{stimulus["question_filename"]}" {event_ids["question"]};\n'
            output += '};\n'
            output += '\n'
            output += 'TEMPLATE "stimulus.tem" {\n'
            output += 'word file code;\n'
    output += '};\n'
    return output, run

with open('run1.sce', 'w') as file:
    output, run = build_run()
    file.write(output)
    print(output)
    run.to_csv('run1.csv')
with open('run2.sce', 'w') as file:
    output, run = build_run()
    file.write(output)
    print(output)
    run.to_csv('run2.csv')
with open('run3.sce', 'w') as file:
    output, run = build_run()
    file.write(output)
    print(output)
    run.to_csv('run3.csv')
with open('practice.sce', 'w') as file:
    output, run = build_run()
    file.write(output)
    print(output)
    run.to_csv('practice.csv')
