from msilib.schema import Error
import numpy as np
import scipy as sp
import scipy.spatial.distance as sp_sd
#from scipy.spatial.distance import pdist
#from scipy.spatial.distance import squareform
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.filters import threshold_otsu

import sys

# adding Folder_2 to the system path
#sys.path.insert(0, 'C:\\Users\\juanf\\OneDrive\\Documents\\GitHub\\Learning-graphs-from-data\\')
import utils.gl.src.gl_models as gl
import matplotlib.pyplot as plt


def remove_intensity_outliers(img):
    x = img.flatten()
    q1 = np.quantile(x,0.25)
    q3 = np.quantile(x,0.75)
    iqr = q3-q1

    return np.minimum( img, q3 + 1.5*iqr)

def prepro_pipeline(dataset,n_spixels,context_radius):

    dataset['after'] = dataset['after']/np.max(dataset['after'].flatten())
    dataset['before'] = dataset['before']/np.max(dataset['before'].flatten())

    if dataset['after'].ndim > 2:
        dataset['after_avg'] = np.mean(dataset['after'], axis = 2)
    if dataset['before'].ndim > 2:
        dataset['before_avg'] = np.mean(dataset['before'], axis = 2)

    rows = dataset['after'].shape[0]; cols = dataset['after'].shape[1]

    # Create a pseudo rgb image for super pixel segmentation
    pseudo_rgb_img = np.zeros((rows,cols,3))
    if not 'after_avg' in dataset.keys():
        pseudo_rgb_img[:,:,0] = dataset['after']
    else:
        pseudo_rgb_img[:,:,0] = dataset['after_avg']
    
    if not 'before_avg' in dataset.keys():
        pseudo_rgb_img[:,:,1] = dataset['before']
    else:
        pseudo_rgb_img[:,:,1] = dataset['before_avg']

    pseudo_rgb_img[:,:,2] = np.abs(pseudo_rgb_img[:,:,0]-pseudo_rgb_img[:,:,1])
    pseudo_rgb_img[:,:,2] = pseudo_rgb_img[:,:,2]/np.max(pseudo_rgb_img[:,:,2].flatten())
 
    segments = slic(pseudo_rgb_img, n_segments=n_spixels, compactness=10, sigma=1,
                       start_label=1)

    thresh = threshold_otsu(pseudo_rgb_img[:,:,2] )
    binary = pseudo_rgb_img[:,:,2] > thresh

    #gradient = sobel(rgb2gray(pseudo_rgb_img))
    #segments = watershed(gradient, markers=1000, compactness=0.001)

    #segments_slic = slic(pseudo_rgb_img[:,:,2], n_segments=n_spixels, compactness=10, sigma=1,
    #                    start_label=1)

    # Generate node features X and one-hot encoded labels Y
    X_mean, X, F = img_2_node_features_matrix(np.dstack((dataset['after'],dataset['before'])), dataset['gt'], segments, context_radius)
    #X_mean, X, F = img_2_node_features_matrix(np.dstack((dataset['after'],dataset['before'])), binary, segments, context_radius)

    return X_mean, X, F, segments

def spixels_upsampling(vector, segments):
    upsampled_image = np.zeros(segments.shape)
    for i in range(vector.size):
        upsampled_image[segments==i+1] = vector[i]
    return upsampled_image

def img_2_node_features_matrix(img, gt, segments, R):
    
    number_of_superpixels = len(np.unique(segments))

    H = img.shape[0]
    W = img.shape[1]
    C = img.shape[2]

    #if context:

        # Estimate smallest radious
        #for i in range(number_of_superpixels):
        #    spixel_i_mask = segments == i+1
        #    subscripts = np.argwhere(spixel_i_mask)
        #    idx_spixel = np.argmin(np.sum((subscripts - np.mean(subscripts,axis=0))**2,axis=1))
        #    if i > 0:
        #        width_i = np.max(subscripts[:,1])-np.min(subscripts[:,1])
        #        height_i = np.max(subscripts[:,0])-np.min(subscripts[:,0])
        #        width = np.min([width_i,width])
        #        height = np.min([height_i,height])
        #    else:
        #        width = np.max(subscripts[:,1])-np.min(subscripts[:,1])
        #        height = np.max(subscripts[:,0])-np.min(subscripts[:,0])
        
        #R = int((np.min([width,height])-1)/2)
        
    X_context = np.zeros((number_of_superpixels,C*(2*R+1)**2))
    X_mean = np.zeros((number_of_superpixels,C))
    Y = np.zeros((number_of_superpixels,2))

    img_padded = np.zeros((H+2*R,W+2*R,C))
    for i in range(C):
        img_padded[:,:,i] = np.pad(img[:,:,i], pad_width = R, mode = 'symmetric')
    segments_padded =np.pad(segments, pad_width = R, mode = 'constant', constant_values = -10)
    gt_padded =np.pad(gt, pad_width = R, mode = 'constant', constant_values = -10)

    #else:
    #    X_context = np.zeros((number_of_superpixels,C*(2*R+1)**2))
    #    X_mean = np.zeros((number_of_superpixels,2))
    #    Y = np.zeros((number_of_superpixels,2))

    #    img_padded = img
    #    segments_padded = segments
    #    gt_padded = gt

    
    for i in range(number_of_superpixels):

        spixel_i_mask = segments_padded == i+1
        # 
        #for j in range(C):
        #    X_mean[i,j] = np.mean(img_padded[spixel_i_mask,j])
        X_mean[i,:] = np.mean(img_padded[spixel_i_mask,:],axis=0)

        #if context:
        subscripts = np.argwhere(spixel_i_mask)
        #idx_spixel = np.argmin(np.sum((subscripts - np.mean(subscripts,axis=0))**2,axis=1))
        idx_spixel = np.argmin(np.sum((img_padded[spixel_i_mask,:] - X_mean[i,:])**2,axis=1)) 
        x0 = subscripts[idx_spixel,1]
        y0 = subscripts[idx_spixel,0]
        context_feature = np.reshape(img_padded[y0-R:y0+R+1,x0-R:x0+R+1,:],(1,C*(2*R+1)**2))
        context_feature = context_feature/np.max(context_feature)
        X_context[i,:] = context_feature



        if np.sum(gt_padded[spixel_i_mask],axis=0) > np.sum(1-gt_padded[spixel_i_mask],axis=0):
            Y[i,:] = [1,0]
        else:
            Y[i,:] = [0,1]
    
    return X_mean, X_context, Y


def construct_sampling_set(label_rate, n, sampling_params = {'seed': 0, 'type': 'random'}, matrix_form = False):
    """ constructs a binary sampling matrix S of shape (nclasses*round(p*min(card(class_i))),n).
    Parameters.
    y, integer encoded class labels
    """
    #seed = sampling_params['seed']
    np.random.seed(sampling_params['seed'])

    if sampling_params['type'] == 'random-class-dependent':
        if not 'labels' in sampling_params.keys():
            raise ValueError('sampling_params["labels"] does not exists. ')

        class_labels, class_cards = np.unique(sampling_params["labels"],return_counts=True)
        nclasses = len(class_labels)
        m = round(label_rate*n/nclasses)

        sample_idx = []
        for c in range(nclasses):
            c_class_idx = np.argwhere(sampling_params["labels"]==class_labels[c]).squeeze()
            c_class_card = class_cards[c]
            
            # Include all samples from classes with less than m elements in them
            if m >= c_class_card:
                m_c = c_class_card
            else:
                m_c = m
    
            random_perm_c_idx = np.argsort(np.random.rand(c_class_card,1),axis=0).squeeze()
            random_perm_c_idx = random_perm_c_idx[0:m_c]
            for i in range(len(random_perm_c_idx)):
                sample_idx.append(c_class_idx[random_perm_c_idx[i]])
            #if c > 0:
            #    sample_idx = np.concatenate((sample_idx.squeeze(), c_class_idx[random_perm_c_idx].squeeze()))
            #else:
            #    sample_idx = c_class_idx[random_perm_c_idx].squeeze()
    elif sampling_params['type'] == 'blue-noise': # random sampling of nodes
        if not 'W' in sampling_params.keys():
            raise ValueError('Adjacency matrix of shape (n,n) must provided in sampling_params["W"].')
        m = round(label_rate*n)
        maxIt = 1000
        sampling_pattern = blue_noise_sampling(sampling_params["W"],m,maxIt,sampling_params['seed'])
        sample_idx = np.arange(0,n)
        sample_idx = sample_idx[sampling_pattern.astype(bool).squeeze()]
    else:
        m = round(label_rate*n)
        sample_idx = np.argsort(np.random.rand(n,1),axis = 0)[0:m]

    if matrix_form:
        I = sp.sparse.eye(n)
        return I.tocsr()[np.asarray(sample_idx).squeeze(),:] 
    else:
        return np.asarray(sample_idx).squeeze()
        

import torch
from torch_cluster import knn_graph

def construct_adj_matrix(X,k,knn_edge_cons = False, k_ = 1):

    d = X.shape[1]
    z = (np.sqrt(d)*sp_sd.pdist(X, 'chebyshev'))**2
    # z = sp_sd.pdist(X, 'chebyshev')
    # z = sp_sd.pdist(X, 'euclidean')
    Z = sp_sd.squareform(z) # turns the condensed form into a n by n distance matrix


    params = {}
    #params['w_0'] = np.zeros((m,m))
    #params['c'] = 1
    if knn_edge_cons:
        edge_index = knn_graph(torch.Tensor(X), k_, loop=False)
        edge_index = edge_index.numpy()
        params['fix_zeros'] = True
        params['edge_mask'] = sp.sparse.coo_matrix((np.ones((edge_index[0].shape)).squeeze(), (edge_index[0],edge_index[1])), shape=Z.shape) #np.zeros(Z.shape) #np.reshape(np.array([1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0]),(16,1))
        params['edge_mask'] = params['edge_mask'].maximum(params['edge_mask'].T)
    params['verbosity'] = 3
    params['maxit'] = 10000
    params['nargout'] = 1

    a = 1
    b = 1
    theta = gl.estimate_theta(Z,k)
    W = gl.gsp_learn_graph_log_degrees(theta*Z,a,b,params)
    W[W<0] = 0
    #W[W<1e-5] = 0
    return W, theta

def blue_noise_sampling(W,s,maxIt,seed):

    n = W.shape[0]

    if n == s:
        return np.ones((n,1))
    # Compute geodesic distance matrix
    K = sp.sparse.csgraph.shortest_path(csgraph=W, directed=False, unweighted=False, method = 'D')
    # 
    sigma = np.std(K.flatten())#-np.std(K.flatten())
    K = np.exp(-(K*K)/(2*(sigma**2)))

    np.random.seed(seed)
    rnd_idx = np.argsort(np.random.rand(n,1),axis=0).squeeze().tolist()

    # Random pattern initialization
    sampling_pattern = np.zeros((n,1))
    sampling_pattern[rnd_idx[0:s]] = 1

    old_idx_tightest_cluster = 1
    old_idx_largest_void = 1
    idx_tightest_cluster = 0
    idx_largest_void = 0

    cntr = 0
    while (idx_largest_void != old_idx_tightest_cluster) or (idx_tightest_cluster!= old_idx_largest_void) or (cntr > maxIt):

        old_idx_tightest_cluster = idx_tightest_cluster
        old_idx_largest_void = idx_largest_void

        nodes_2_samples_density = np.sum(K[:,np.argwhere(sampling_pattern.squeeze()).squeeze()], axis = 1)

        idx_clusters = np.argwhere(sampling_pattern.squeeze())
        idx_voids = np.argwhere(1 - sampling_pattern.squeeze())

        idx_tightest_cluster =  idx_clusters[np.argmax(nodes_2_samples_density[idx_clusters],axis=0)].squeeze()
        idx_largest_void =  idx_voids[np.argmin(nodes_2_samples_density[idx_voids],axis=0)].squeeze()   

        # Swap tightest cluster sample with largest void sample (unselected node)

        sampling_pattern[idx_tightest_cluster] = 0
        sampling_pattern[idx_largest_void] = 1

        cntr = cntr + 1

    return sampling_pattern

def kappa_coeff(y,y_hat):
    #confusion_matrix = np.zeros((2,2))
    TP = np.sum(np.logical_and(y,y_hat)).astype(np.float32)
    FP = np.sum(np.logical_and(np.logical_not(y),y_hat)).astype(np.float32)
    FN = np.sum(np.logical_and(y,np.logical_not(y_hat))).astype(np.float32)
    TN = np.sum(np.logical_and(np.logical_not(y),np.logical_not(y_hat))).astype(np.float32)
    #2*(TP*TN-FN*FP)/((TP+FP)*(FP+TN)+(TP+FN)*(FN+TN))
    return 2*((TP*TN)-(FN*FP))/((TP+FP)*(FP+TN)+(TP+FN)*(FN+TN))

import os.path
from os import path

def save_figure(path_to_file):
    cntr = 0
    while os.path.isfile(path_to_file):
        cntr = cntr + 1
        idx_start = path_to_file.find('_v_')
        path_to_file = path_to_file[0:idx_start] + '_v_' + str(cntr) + '.png' 
    
    plt.savefig(path_to_file, dpi=600, bbox_inches='tight')
