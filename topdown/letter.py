import matplotlib.pyplot as plt
from matplotlib import cm

"""
 _ _       1     2    
|\|/|   3  4  5  6  7
 - -       8     9
|/|\|  10 11 12 13 14
 - -      15    16
"""
letters = dict(
    #    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
    a = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    b = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1],
    c = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
    d = [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1],
    e = [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1],
    f = [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    g = [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
    h = [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    i = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    j = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1],
    k = [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
    l = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
    m = [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    n = [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    o = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1],
    p = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    q = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1],
    r = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
    s = [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    t = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    u = [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1],
    v = [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    w = [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
    x = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    y = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    z = [1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
)

def plot_letter(letter, ax=None, vmax=1.0, cmap='gray_r'):
    cmap = cm.get_cmap(cmap)
    if ax is None:
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.set_axis_off()
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)

    if letter[0]:
        ax.plot([1, 2], [3, 3], color=cmap(letter[0] / vmax))
    if letter[1]:
        ax.plot([2, 3], [3, 3], color=cmap(letter[1] / vmax))
    if letter[2]:
        ax.plot([1, 1], [2, 3], color=cmap(letter[2] / vmax))
    if letter[3]:
        ax.plot([2, 1], [2, 3], color=cmap(letter[3] / vmax))
    if letter[4]:
        ax.plot([2, 2], [2, 3], color=cmap(letter[4] / vmax))
    if letter[5]:
        ax.plot([2, 3], [2, 3], color=cmap(letter[5] / vmax))
    if letter[6]:
        ax.plot([3, 3], [2, 3], color=cmap(letter[6] / vmax))
    if letter[7]:
        ax.plot([1, 2], [2, 2], color=cmap(letter[7] / vmax))
    if letter[8]:
        ax.plot([2, 3], [2, 2], color=cmap(letter[8] / vmax))
    if letter[9]:
        ax.plot([1, 1], [1, 2], color=cmap(letter[9] / vmax))
    if letter[10]:
        ax.plot([1, 2], [1, 2], color=cmap(letter[10] / vmax))
    if letter[11]:
        ax.plot([2, 2], [1, 2], color=cmap(letter[11] / vmax))
    if letter[12]:
        ax.plot([2, 3], [2, 1], color=cmap(letter[12] / vmax))
    if letter[13]:
        ax.plot([3, 3], [1, 2], color=cmap(letter[13] / vmax))
    if letter[14]:
        ax.plot([1, 2], [1, 1], color=cmap(letter[14] / vmax))
    if letter[15]:
        ax.plot([2, 3], [1, 1], color=cmap(letter[15] / vmax))
