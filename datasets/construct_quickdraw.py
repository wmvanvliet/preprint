import quickdraw
from numpy.random import RandomState
import os
from tqdm import tqdm
import warnings

warnings.simplefilter('ignore')

rnd = RandomState(0)
names = rnd.choice(quickdraw.names.QUICK_DRAWING_NAMES, 200)

pbar = tqdm(total=200*550)
for name in names:
    dirname = name.replace(' ', '_')
    os.makedirs('/l/vanvlm1/quickdraw/train/%s' % dirname, exist_ok=True)
    os.makedirs('/l/vanvlm1/quickdraw/val/%s' % dirname, exist_ok=True)

    group = quickdraw.QuickDrawDataGroup(name, max_drawings=550,
                                         recognized=True, print_messages=False)
    for i, image in enumerate(group.drawings):
        image.image.thumbnail((64, 64))
        if i < 500:
            image.image.save('/l/vanvlm1/quickdraw/train/%s/%03d.JPEG' % (dirname, i))
        else:
            image.image.save('/l/vanvlm1/quickdraw/val/%s/%03d.JPEG' % (dirname, i))
        pbar.update(1)
pbar.close()
