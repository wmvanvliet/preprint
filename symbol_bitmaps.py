"""
Create pickle object containing list of 2d numpy arrays of the symbols used in the Epasana study. 
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import pickle

path = '/m/nbe/scratch/epasana/stimuli/'

pic = plt.imread(path+'symbolit1.tif')
pic2 = plt.imread(path+'symbolit8.tif')
pic3 = plt.imread(path+'symbolit4.tif')

df = pd.DataFrame(pic)
df2 = pd.DataFrame(pic2)
df3 = pd.DataFrame(pic3)

df = df.loc[range(38,61),range(33,168)]
df2 = df2.loc[range(38,61),range(33,168)]
df3 = df3.loc[range(38,61),range(33,168)]

star = df.loc[range(40,60),range(34,51)]
triangle = df.loc[range(40,60),range(51,67)]
circle = df.loc[range(40,60),range(67,84)]
symb = df.loc[range(40,60),range(86,100)]
star2 = df.loc[range(40,60),range(101,116)]
box = df2.loc[range(40,60),range(38,52)]
slash = df2.loc[range(40,60),range(53,63)]
hexa = df2.loc[range(40,60),range(95,111)]
spiral = df2.loc[range(40,60),range(110,128)]
diamond = df3.loc[range(40,60),range(62,79)]
angle = df3.loc[range(40,60),range(110,129)]

symbols = [star,triangle,circle,symb,star2,box,slash,hexa,spiral,diamond,angle]
npsymbols = []

for item in symbols:
    #hmap=sb.heatmap(item)
    #plt.show()
    npsymbols.append(item.to_numpy())

with open('symbol-bitmaps', 'wb') as output:
    pickle.dump(npsymbols, output)







