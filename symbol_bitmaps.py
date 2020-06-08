"""
Create pickle object containing list of 2d numpy arrays of the symbols used in the "epasana" study. 
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

path = '/m/nbe/scratch/epasana/stimuli/'
# We are going to extract the individual symbols from these three stimulus examples.
# Together, they contain every symbol used in the study.
pic = plt.imread(f'{path}/symbolit1.tif')
pic2 = plt.imread(path+'symbolit8.tif')
pic3 = plt.imread(path+'symbolit4.tif')

star = pic[40:60,34:51]
triangle = pic[40:60,51:67]
circle = pic[40:60,67:84]
symb = pic[40:60,86:100]
star2 = pic[40:60,101:116]
box = pic2[40:60,38:52]
slash = pic2[40:60,53:63]
hexa = pic2[40:60,95:111]
spiral = pic2[40:60,110:128]
diamond = pic3[40:60,62:79]
angle = pic3[40:60,110:129]

symbols = [star,triangle,circle,symb,star2,box,slash,hexa,spiral,diamond,angle]

with open('/m/nbe/scratch/reading_models/datasets/symbol-bitmaps', 'wb') as output:
    pickle.dump(symbols, output)





