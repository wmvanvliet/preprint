import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.spatial as ss
import numpy as np
import mne
import mne_rsa

def make_indexlist(stim_order, names):
    indexlist = []
    for name in stim_order:
        indexlist.append(names.index(name))
    indexlist = np.array(indexlist)
    return indexlist

def load_images_from_folder(folder):
    images = []
    names = []
    for count, filename in enumerate(sorted(os.listdir(folder)), start=1):
        if (filename != 'Thumbs.db') & (filename != 'stimulus_text.csv'):
            names.append(filename)
            img = plt.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
    return images, names

def compute_rsa(epofile, orig_dsm, names, epo_dsm_metric, rsa_metric, spat_rad, temp_rad, interval):
    epochs = mne.read_epochs(epofile)
    epochs = epochs.resample(100)
    epochs = epochs.pick_types(meg='grad')                                                                                     
    stim_order = epochs.metadata['tif_file']
    y = make_indexlist(stim_order, names)
    dsm = pd.DataFrame(orig_dsm,columns=np.arange(0,len(names)))
    for i in range(len(names)):
        if i not in y:
            dsm = dsm.drop(i)
            dsm = dsm.drop(i,axis=1)
    dsm = dsm.to_numpy()
    rsa_result = mne_rsa.rsa_epochs(
        epochs,                          
        dsm,                      
        y = y,
        epochs_dsm_metric=epo_dsm_metric,     # Metric to compute the EEG DSMs
        rsa_metric=rsa_metric,                # Metric to compare model and EEG DSMs
        spatial_radius=spat_rad,              # Spatial radius of the searchlight patch
        temporal_radius=temp_rad,             # Temporal radius of the searchlight path
        tmin=interval[0], tmax=interval[1],   # To save time, only analyze this time interval
        n_jobs=1,                             # Only use one CPU core. Increase this for more speed.
        verbose=True)                         # Set to True to display a progress bar
                                                                                                                                                                                              
    return rsa_result
            

def main():
   
    # path to /m/nbe/scratch/epasana
    path = '/m/nbe/scratch/epasana' 

    images, names = load_images_from_folder(path+'/stimuli')
    orig_dsm = mne_rsa.compute_dsm(images,'euclidean')
    orig_dsm = ss.distance.squareform(orig_dsm)
    subs = ['sub01-epo.fif','sub02-epo.fif','sub03-epo.fif','sub04-epo.fif','sub05-epo.fif','sub06-epo.fif','sub07-epo.fif','sub08-epo.fif','sub10-epo.fif','sub11-epo.fif']

    evokeds = []

    for sub in subs:
        epo_path = path+'/marijn/'+sub
        rsa = compute_rsa(epo_path, orig_dsm, names,'correlation','spearman',0.02,0.05,[0,0.5])
        evokeds.append(rsa)

    grand = mne.grand_average(evokeds)
    grand.plot_topo()

    pass

if __name__ == "__main__":
    main()
