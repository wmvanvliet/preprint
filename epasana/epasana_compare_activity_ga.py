import mne
from collections import defaultdict
from tqdm import tqdm

from config import fname

act = defaultdict(list)
for subject in tqdm(range(1, 16)):
    ev = mne.read_evokeds(fname.layer_corr(subject=subject), verbose=False)
    for e in ev:
        act[e.comment].append(e)
for cond, evokeds in act.items():
    act[cond] = mne.grand_average(evokeds)
    act[cond].comment = cond
