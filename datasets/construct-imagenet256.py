# encoding: utf-8
"""
Construct a dataset containing 256x256 pixel images of imagenet words. Uses the
same 200 classes used in the "tiny-imagenet" dataset.
"""
#
import argparse
from glob import glob
from PIL import Image
import tfrecord
import pandas as pd
from io import BytesIO
import numpy as np
from torchvision.transforms import Resize, CenterCrop
from os import makedirs
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate the imagnet256 dataset')
parser.add_argument('path', type=str, help='The path to write the dataset to.')
parser.add_argument('set', type=str, help='Specify either "train" to generate the training set and to "test" to generate the test set.')
args = parser.parse_args()

# Set this to where-ever the original imaganet training set can be found
imagenet_path = '/l/vanvlm1/imagenet_train'

wnids = ['n02124075', 'n04067472', 'n04540053', 'n04099969', 'n07749582',
         'n01641577', 'n02802426', 'n09246464', 'n07920052', 'n03970156',
         'n03891332', 'n02106662', 'n03201208', 'n02279972', 'n02132136',
         'n04146614', 'n07873807', 'n02364673', 'n04507155', 'n03854065',
         'n03838899', 'n03733131', 'n01443537', 'n07875152', 'n03544143',
         'n09428293', 'n03085013', 'n02437312', 'n07614500', 'n03804744',
         'n04265275', 'n02963159', 'n02486410', 'n01944390', 'n09256479',
         'n02058221', 'n04275548', 'n02321529', 'n02769748', 'n02099712',
         'n07695742', 'n02056570', 'n02281406', 'n01774750', 'n02509815',
         'n03983396', 'n07753592', 'n04254777', 'n02233338', 'n04008634',
         'n02823428', 'n02236044', 'n03393912', 'n07583066', 'n04074963',
         'n01629819', 'n09332890', 'n02481823', 'n03902125', 'n03404251',
         'n09193705', 'n03637318', 'n04456115', 'n02666196', 'n03796401',
         'n02795169', 'n02123045', 'n01855672', 'n01882714', 'n02917067',
         'n02988304', 'n04398044', 'n02843684', 'n02423022', 'n02669723',
         'n04465501', 'n02165456', 'n03770439', 'n02099601', 'n04486054',
         'n02950826', 'n03814639', 'n04259630', 'n03424325', 'n02948072',
         'n03179701', 'n03400231', 'n02206856', 'n03160309', 'n01984695',
         'n03977966', 'n03584254', 'n04023962', 'n02814860', 'n01910747',
         'n04596742', 'n03992509', 'n04133789', 'n03937543', 'n02927161',
         'n01945685', 'n02395406', 'n02125311', 'n03126707', 'n04532106',
         'n02268443', 'n02977058', 'n07734744', 'n03599486', 'n04562935',
         'n03014705', 'n04251144', 'n04356056', 'n02190166', 'n03670208',
         'n02002724', 'n02074367', 'n04285008', 'n04560804', 'n04366367',
         'n02403003', 'n07615774', 'n04501370', 'n03026506', 'n02906734',
         'n01770393', 'n04597913', 'n03930313', 'n04118538', 'n04179913',
         'n04311004', 'n02123394', 'n04070727', 'n02793495', 'n02730930',
         'n02094433', 'n04371430', 'n04328186', 'n03649909', 'n04417672',
         'n03388043', 'n01774384', 'n02837789', 'n07579787', 'n04399382',
         'n02791270', 'n03089624', 'n02814533', 'n04149813', 'n07747607',
         'n03355925', 'n01983481', 'n04487081', 'n03250847', 'n03255030',
         'n02892201', 'n02883205', 'n03100240', 'n02415577', 'n02480495',
         'n01698640', 'n01784675', 'n04376876', 'n03444034', 'n01917289',
         'n01950731', 'n03042490', 'n07711569', 'n04532670', 'n03763968',
         'n07768694', 'n02999410', 'n03617480', 'n06596364', 'n01768244',
         'n02410509', 'n03976657', 'n01742172', 'n03980874', 'n02808440',
         'n02226429', 'n02231487', 'n02085620', 'n01644900', 'n02129165',
         'n02699494', 'n03837869', 'n02815834', 'n07720875', 'n02788148',
         'n02909870', 'n03706229', 'n07871810', 'n03447447', 'n02113799',
         'n12267677', 'n03662601', 'n02841315', 'n07715103', 'n02504458']

n = 500 if args.set == 'train' else 50
labels = np.zeros(len(wnids) * n, dtype=np.int)

makedirs(args.path, exist_ok=True)
writer = tfrecord.TFRecordWriter(f'{args.path}/{args.set}.tfrecord')

for label, wnid in enumerate(tqdm(wnids)):
    image_fnames = glob(f'{imagenet_path}/{wnid}/*.JPEG')
    chosen_fnames = np.random.choice(image_fnames, size=n, replace=False)
    for i, fname in enumerate(chosen_fnames):
        img = Image.open(fname)
        img = Resize(256)(img)
        img = CenterCrop(256)(img)

        buf = BytesIO()
        img.save(buf, format='jpeg')

        writer.write({
            'image/height': (256, 'int'),
            'image/width': (256, 'int'),
            'image/colorspace': (b'RGB', 'byte'),
            'image/channels': (3, 'int'),
            'image/class/label': (label, 'int'),
            'image/class/wnid': (wnid.encode('utf-8'), 'byte'),
            'image/format': (b'JPEG', 'byte'),
            'image/filename': (f'{wnid}.JPEG'.encode('utf-8'), 'byte'),
            'image/encoded': (buf.getvalue(), 'byte'),
        })

        labels[label * n + i] = label
writer.close()

tfrecord.tools.create_index(f'{args.path}/{args.set}.tfrecord', f'{args.path}/{args.set}.index')

# Write metadata and word2vec vectors
df = pd.DataFrame(dict(text=np.repeat(wnids, n), label=np.repeat(np.arange(len(wnids)), n)))
df.to_csv(f'{args.path}/{args.set}.csv')

# FIXME: use real vectors instead of zeros!
pd.DataFrame(np.zeros((len(wnids), 300)), index=np.zeros(len(wnids))).to_csv(f'{args.path}/vectors.csv')
