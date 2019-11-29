import pandas as pd
from shutil import copyfile
from os import makedirs
from os.path import basename
from glob import glob
from PIL import Image

contents = pd.read_csv('/l/vanvlm1/facescrub/contents.txt', sep=' ', header=None)
contents.columns = ['dir', 'n']
contents = contents.sort_values('n').tail(200)
contents = contents.reset_index(drop=True)
contents['label'] = contents['dir'].str[2:-5]

n_val = 10
for _, row in contents.iterrows():
    print(row.label, flush=True)
    makedirs(f'/l/vanvlm1/facescrub/train/{row.label}', exist_ok=True)
    makedirs(f'/l/vanvlm1/facescrub/val/{row.label}', exist_ok=True)
    for i, fname in enumerate(glob(f'/l/vanvlm1/facescrub/{row.dir}/*.jpg')):
        try:
            im = Image.open(fname)
            im.thumbnail((64, 64))
            if i < n_val:
                im.save('/l/vanvlm1/facescrub/val/%s/%s' % (row.label, basename(fname)))
            else:
                im.save('/l/vanvlm1/facescrub/train/%s/%s' % (row.label, basename(fname)))
        except:
            print('   fail')
            pass
