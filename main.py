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
#import models.gcd_models as gcd_models
import experiments.gcd_experiments as gcd_experiments


list_dataset_names = ['Alaska','California','Atlantico']

for dataset_idx in range(3):
    if dataset_idx == 0:
        dataset = sio.loadmat(datasets_dir + '\\Alaska_dataset.mat')
    elif dataset_idx == 1: 
        dataset = sio.loadmat(datasets_dir + '\\California_dataset.mat')
    elif dataset_idx == 2: 
        dataset = sio.loadmat(datasets_dir + '\\Atlantico_dataset.mat')
        dataset['after'] = np.minimum(dataset['after'],0.9)
        dataset['before'] = np.minimum(dataset['before'],0.9)

    dataset_name = list_dataset_names[dataset_idx]


    n_spixels = 1500
    context_radius = 5
    X_mean, X, F, segments_slic = gcd_utils.prepro_pipeline(dataset,n_spixels,context_radius)

    #plt.figure()
    #ax0 = plt.gca()
    #im = ax0.imshow(mark_boundaries(dataset['gt'].astype(float), segments_slic))
    # Construct graph


    n = X.shape[0]
    f_ask = dataset['gt']

    k_list = [3, int(n/16), int(n/4)]
    p = 0.3 # fraction of positive labels used for training
    segments = segments_slic
    stype='random-class-dependent'
    gcd_experiments.exp_effect_of_theta(k_list, p, dataset_name, f_ask, F, X_mean, X, segments,stype)

