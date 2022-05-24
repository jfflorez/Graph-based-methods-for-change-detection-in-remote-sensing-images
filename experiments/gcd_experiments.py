import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy as sp

import utils.gcd_utils as gcd_utils
import models.gcd_models as gcd_models

from scipy.sparse import coo_matrix, hstack, vstack, find
from scipy.sparse.linalg import spsolve

def exp_kappa_vs_label_rate(F_ask,F,X,segments,W,stype):

    # Construct a sampling matrix of shape (k,n)

    n = len(np.unique(segments)) # number of nodes (superpixels)
    if not n == W.shape[0]:
        raise ValueError('W shape must be (n,n).')

    if stype == 'random-class-dependent':
        k_max = np.min([np.sum(F[:,0]),np.sum(F[:,1])])
    else:
        k_max = n
    k = np.linspace(1,k_max,num=10).astype(np.int32)
    label_rate = np.zeros(k.shape)
    kappa = np.zeros((k.size,2))
    f = 1-np.argmax(F,axis=1)
    seed = 100
    for i in range(len(k)):
        print('iteration count: ', i,'out of ', len(k))

        #k = int(label_rate[i]*n)
        S = gcd_utils.construct_sampling_matrix(k[i],n,stype,seed,F)

        label_rate[i] = S.shape[0]/n

        f_hat = gcd_models.gsm_estimation(S,F,W)  
        kappa[i,0] = gcd_utils.kappa_coeff(F_ask,gcd_utils.spixels_upsampling(f_hat,segments))
        f_hat = gcd_models.gcn_estimation(S,F,W,X)
        kappa[i,1] = gcd_utils.kappa_coeff(F_ask,gcd_utils.spixels_upsampling(f_hat,segments))

    kappa_limit = gcd_utils.kappa_coeff(F_ask,
                                        gcd_utils.spixels_upsampling(f,segments))
    plt.figure()
    ax0 = plt.gca()
    ax0.plot(label_rate,kappa[:,0], label = 'gsm (jimenez at al)')
    ax0.plot(label_rate,kappa[:,1], label = 'gcn')
    ax0.plot(label_rate,kappa_limit*np.ones(label_rate.shape), label = 'fundamental limit')
    ax0.legend()
    return plt.gcf



def exp_effect_of_theta(k_list, p, dataset_name, f_ask,F,X_mean, X,segments,stype):

    """ 
    Parameters:
    p : float, value between 0 and 1, which denotes the fraction of minority class labels to be taken from each class for training."""

    n = X_mean.shape[0] # number of nodes        
    n_pos = np.sum(F[:,0])
    n_neg = np.sum(F[:,1])
    if not n_neg + n_pos == n:
        raise ValueError('The number of positive and negative examples must add up to the number of nodes n.')
    

    # create sampling matrix that takes k labels per class uniformly at random
    m = int(round(p*n_pos))
    seed = 0
    stype='random-class-dependent'
    #stype = 'random'
    S = gcd_utils.construct_sampling_matrix(m,n,stype,seed,F)


    cd_methods = ['gsm', 'gcn']

    fig, ax = plt.subplots(len(cd_methods)+1, len(k_list), figsize=(10, 10), sharex=True, sharey=True)

    

    for k_idx in range(len(k_list)):
        # Learn graph's adj mtx using only spixels mean value. This seems to be optimal both for gsm and gcn

        W, theta = gcd_utils.construct_adj_matrix(X_mean, k = k_list[k_idx], knn_edge_cons=True, k_ = 2*k_list[k_idx])
        W = W/np.max(W[W>0])   

        for m_idx in range(len(cd_methods)):
                if cd_methods[m_idx] == 'gsm':
                    f_hat = gcd_models.gsm_estimation(S,F,W)
                elif cd_methods[m_idx] == 'gcn':
                    n_hidden_channels = 16#X.shape[1]
                    f_hat = gcd_models.gcn_estimation(S,F,W,X,n_hidden_channels)
            
                c_hat = gcd_utils.spixels_upsampling( f_hat, segments )
                kappa_c_hat = gcd_utils.kappa_coeff( f_ask.flatten().astype(bool), c_hat.flatten().astype(bool) )

                ax[m_idx, k_idx].imshow(c_hat)
                ax[m_idx, k_idx].set_xlabel('\u03BA='+str(np.round(kappa_c_hat,3)))
                #ax[m_idx, theta_idx].set_title(r'$\theta$'+'=' + str(theta_list[theta_idx]) +'('+'\u03BA='+str(np.round(kappa_c_hat,3))+')')
                if k_idx == 0:
                    ax[m_idx, k_idx].set_ylabel(cd_methods[m_idx])
                if m_idx == 0:
                    ax[m_idx, k_idx].set_title(r'$\theta$'+'=' + str(np.round(theta,3)) + ',\n  $k$='  + str(k_list[k_idx]))
                ax[m_idx, k_idx].set_xticks([])
                ax[m_idx, k_idx].set_yticks([])

    for k_idx in range(len(k_list)):
        if k_idx == 1:
            ax[len(cd_methods), k_idx].imshow(f_ask)
            ax[len(cd_methods), k_idx].set_ylabel('Ground truth')
        ax[len(cd_methods), k_idx].set_axis_off()             
      
    plt.savefig( "./figs\\"+ "fig_exp_effect_of_theta_"+ dataset_name +"111.png", dpi=600, bbox_inches='tight')

#    plt.savefig( ".>figs\\"+ "fig_2_Exp1_CaseStudy"+str(int(case_study))+"_accuracy.png", dpi=600,
#    bbox_extra_artists=(lgd,),
#    bbox_inches='tight')

        



