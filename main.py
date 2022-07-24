import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

#from skimage.util import numpy_pad
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

import scipy.io as sio
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

import sys, os 
dirname = os.path.dirname(__file__)
datasets_dir = os.path.join(dirname, 'datasets')

import utils.gl.src.gl_models as gl
import utils.gcd_utils as gcd_utils 
import experiments.gcd_experiments as gcd_experiments


list_dataset_names = ['Alaska','California','Atlantico']


for dataset_idx in range(0,3):

    # Load dataset
    if dataset_idx == 0:
        dataset = sio.loadmat(datasets_dir + '\\Alaska_dataset.mat')
    elif dataset_idx == 1: 
        dataset = sio.loadmat(datasets_dir + '\\California_dataset.mat')
    elif dataset_idx == 2: 
        dataset = sio.loadmat(datasets_dir + '\\Atlantico_dataset.mat')
        dataset['after'] = np.minimum(dataset['after'],0.9)
        dataset['before'] = np.minimum(dataset['before'],0.9)
   
    dataset_name = list_dataset_names[dataset_idx]

    # Create dataset from image pairs based on slic superpixels
    n_spixels = 1500
    context_radius = 5
    X_mean, X, F, segments_slic = gcd_utils.prepro_pipeline(dataset,n_spixels,context_radius)

    if dataset_idx == 1:
        fig, ax = plt.subplots(1, 3, figsize=(10, 10), sharex=True, sharey=True)
        ax[0].imshow(dataset['before'])
        ax[0].set_title('  Lansat-8, \n NIR image')
        ax[0].set_xlabel('Jan 5, 2017')
        ax[1].imshow(dataset['after'])
        ax[1].set_xlabel('Feb 18, 2017')
        ax[1].set_title('  Sentinel-1, \n SAR image')
        ax[2].imshow(dataset['gt'])
        ax[2].set_title('Change Map')
        for i in range(3):
            ax[i].set_xticks([])
            ax[i].set_yticks([])   
        plt.savefig( "./figs\\"+ "fig_dataset_"+ dataset_name +".png", dpi=600, bbox_inches='tight')
    

    n = X.shape[0]
    f_ask = dataset['gt']
    segments = segments_slic
    k_list = [int(n/32), int(n/4)]

    exp_number = 1
    if exp_number == 0:
    
        p = 0.2 # fraction of positive labels used for training
        
        stype = 'random-class-dependent'
        #stype= 'random'
        #stype= 'blue-noise'
        gcd_experiments.exp_effect_of_theta(k_list, p, dataset_name, f_ask, F, X_mean, X, segments,stype)
    elif exp_number == 1:
        stype = ['random-class-dependent', 'random', 'blue-noise']
        gcd_experiments.exp_00_kappa_vs_label_rate(f_ask,F,X,X_mean,segments,k_list[0],stype,dataset_name)
    
    #gcd_experiments.exp_effect_of_random_sampling(k_list, p, dataset_name, f_ask,F,X_mean, X,segments,stype)
    #nclasses_ss = 7
    #gcd_experiments.exp_accuracy_vs_ss_level(k_list[0], p, nclasses_ss, dataset_name, f_ask, F, X_mean, X,segments,stype)
    #gcd_experiments.exp_accuracy_vs_reg_level(k_list[0], p, dataset_name, f_ask, F, X_mean, X,segments,stype)
    