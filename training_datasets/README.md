These scripts generate training images for the computational models. Each dataset is around 100_000 images. To prevent folders with 100_000 files in them, the images are saved as TensorFlow Records. The [`dataloaders.py`](../../../blob/master/dataloaders.py) file contains PyTorch dataloaders to load these.

| script                          | dataset it generates
|---------------------------------|---------------------------------------------------------------------
| construct_epasana-10kwords.py   | The valid Finnish words used in "Epasana", plus common Finnish words
| construct_epasana-consonants.py | Random consonant strings like the ones used in "Epasana"
| construct_epasana-symbols.py    | Random symbol strings like the ones used in "Epasana"
| construct-imagenet256.py        | A selection of the ImageNet database
| construct_noise.py              | Images containing only visual noise
| construct-pilot-nontext.py      | Random consonant and symbol strings like the ones used in the pilot study
| construct-pilot-words.py        | The words used in the pilot study
