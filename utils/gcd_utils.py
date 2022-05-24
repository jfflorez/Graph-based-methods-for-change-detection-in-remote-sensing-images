import numpy as np
import scipy as sp
import scipy.spatial.distance as sp_sd
#from scipy.spatial.distance import pdist
#from scipy.spatial.distance import squareform
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.color import rgb2gray
from skimage.filters import sobel

import sys

# adding Folder_2 to the system path
#sys.path.insert(0, 'C:\\Users\\juanf\\OneDrive\\Documents\\GitHub\\Learning-graphs-from-data\\')
import utils.gl.src.gl_models as gl


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
    #gradient = sobel(rgb2gray(pseudo_rgb_img))
    #segments = watershed(gradient, markers=1000, compactness=0.001)

    #segments_slic = slic(pseudo_rgb_img[:,:,2], n_segments=n_spixels, compactness=10, sigma=1,
    #                    start_label=1)

    # Generate node features X and one-hot encoded labels Y
    X_mean, X, F = img_2_node_features_matrix(np.dstack((dataset['after'],dataset['before'])), dataset['gt'], segments, context_radius)

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
        X_context[i,:] = context_feature



        if np.sum(gt_padded[spixel_i_mask],axis=0) > np.sum(1-gt_padded[spixel_i_mask],axis=0):
            Y[i,:] = [1,0]
        else:
            Y[i,:] = [0,1]
    
    return X_mean, X_context, Y


def construct_sampling_matrix(m,n,stype,seed,Y):
    """ constructs a binary sampling matrix S of shape (k,n).
    Parameters.
    Y, one hot encoded class labels
    """
    I = sp.sparse.eye(n)
    np.random.seed(seed)
    if stype == 'random-class-dependent':
        for c in range(Y.shape[1]):
            c_class_idx = np.argwhere(Y[:,c].squeeze()).squeeze()
            random_perm_c_idx = np.argsort(np.random.rand(len(c_class_idx),1),axis=0).squeeze()
            if c > 0:
                sample_idx = np.concatenate((sample_idx, c_class_idx[random_perm_c_idx[0:m]]))
            else:
                sample_idx = c_class_idx[random_perm_c_idx[0:m]]
    else: # random sampling of nodes
        sample_idx = np.argsort(np.random.rand(n,1),axis = 0)[0:m]
    
    return I.tocsr()[sample_idx.squeeze(),:]

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
        params['edge_mask'] = params['edge_mask'] + params['edge_mask'].T
    params['verbosity'] = 3
    params['maxit'] = 5000
    params['nargout'] = 1

    a = 1
    b = 1
    theta = gl.estimate_theta(Z,k)
    W = gl.gsp_learn_graph_log_degrees(theta*Z,a,b,params)
    W[W<1e-5] = 0
    return W, theta

def kappa_coeff(y,y_hat):
    #confusion_matrix = np.zeros((2,2))
    TP = np.sum(np.logical_and(y,y_hat)).astype(np.float32)
    FP = np.sum(np.logical_and(np.logical_not(y),y_hat)).astype(np.float32)
    FN = np.sum(np.logical_and(y,np.logical_not(y_hat))).astype(np.float32)
    TN = np.sum(np.logical_and(np.logical_not(y),np.logical_not(y_hat))).astype(np.float32)
    #2*(TP*TN-FN*FP)/((TP+FP)*(FP+TN)+(TP+FN)*(FN+TN))
    return 2*((TP*TN)-(FN*FP))/((TP+FP)*(FP+TN)+(TP+FN)*(FN+TN))
