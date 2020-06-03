These scripts generate training images for the computational models. Each dataset is around 100_000 images. To prevent folders with 100_000 files in them, the images are saved as pickled python lists of PNG byte-encoded images. The [`dataloaders.py`](../../../blob/master/dataloaders.py) file contains PyTorch dataloaders to load these pickled files.

| script                        | dataset it generates
|-------------------------------|---------------------------------------------------------------------
| construct_10k-words.py        | 10000 short finnish words
| construct_facescrub.py        | images of celebrity faces
| construct_quickdraw.py        | doodles downloaded from https://quickdraw.withgoogle.com
| construct_redness1_dataset.py | words used in the Redness1 experiment
| construct_redness2_dataset.py | words used in the Redness2 experiment
| construct_tiny-consonants.py  | random consonant strings
| construct_tiny-symbols.py     | random symbol strings
| construct_tiny-text-noise.py  | random consonant+symbol strings
| construct_tiny-words.py       | words used in the pilot experiment and some additional finnish words
| construct_words.py            | 128x128 pixel images of words used in Epasana experiment
| construct_consonants.py       | 128x128 pixel images of consonant strings
| construct_symbols.py          | 128x128 pixel images of symbol strings
