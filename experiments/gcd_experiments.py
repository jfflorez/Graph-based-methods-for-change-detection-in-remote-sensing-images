from cmath import inf
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy as sp

import utils.gcd_utils as gcd_utils
import models.gcd_models as gcd_models

from scipy.sparse import coo_matrix, hstack, vstack, find
from scipy.sparse.linalg import spsolve


def exp_00_kappa_vs_label_rate(F_ask, F, X, X_mean, segments, k, stype, dataset_name):

    """ This script takes a remote sensing image data set created by our function
    
    stype (list): list of strings, each denoting how to sample the available labels for
    semisupervised learning. All avaiable types can be selected by setting 
    stype = ['random class-dependent', 'random', 'blue-noise']
    """

    # Construct a sampling matrix of shape (k,n)
    W, theta = gcd_utils.construct_adj_matrix(X_mean, k, knn_edge_cons=True, k_ = 2*k)
    W = W/np.max(W[W>0])  

    n = len(np.unique(segments)) # number of nodes (superpixels)
    nclasses = F.shape[1]
    f = 1-np.argmax(F,axis=1)

    if not n == W.shape[0]:
        raise ValueError('W shape must be (n,n).')

    
    fig, ax = plt.subplots(len(stype),1)

    for stype_idx in range(len(stype)):

        # Define general sampling set parameters
        sampling_params = {'seed': 0, 'type': stype[stype_idx]}

        # Define type-specific sampling set parameters
        if stype[stype_idx] == 'random-class-dependent':
            k_max = np.min([np.sum(F[:,0]),np.sum(F[:,1])])
            k = np.linspace(1,k_max,num=10).astype(np.int32)
            label_rate = (k*nclasses)/n
            sampling_params['labels'] = f
            
        elif stype[stype_idx] == 'blue-noise':
            k_max = n
            k = np.linspace(2,k_max,num=10).astype(np.int32)
            label_rate = k/n
            D_sq_inv = sp.sparse.spdiags(np.power(np.sum(W,1),-0.5).squeeze(),0, W.shape[0], W.shape[1])
            sampling_params['W'] =  D_sq_inv @ (W @ D_sq_inv)
        else:
            k_max = n
            k = np.linspace(2,k_max,num=10).astype(np.int32)
            label_rate = k/n
        
        kappa = np.zeros((k.size,3))    
        ss_params = {'type': 'without'}
        for i in range(len(k)):
            print('iteration count: ', i,'out of ', len(k))

            S = gcd_utils.construct_sampling_set(label_rate[i], n, sampling_params, matrix_form = True)            

            f_hat0 = gcd_models.gsm_estimation(S,F,W)  
            f_hat1, history, model = gcd_models.gcn_estimation(S,F,W,X,hidden_chs=16, ss_params = ss_params)
            f_hat2, history, model = gcd_models.sgcn_estimation(S, F, W, X, K=2, ss_params = {'type': 'without'})
            
            for m_idx in range(3):
                c_hat = gcd_utils.spixels_upsampling(eval('f_hat'+str(m_idx)), segments )
                kappa[i,m_idx] = gcd_utils.kappa_coeff(F_ask.flatten().astype(bool), c_hat.flatten().astype(bool) ) #gcd_utils.kappa_coeff(F_ask,gcd_utils.spixels_upsampling(f_hat,segments))
          
            #if i == 0:
                # Specify a path to save to            
            #    PATH = "./pretrained_models\\" + "gcn_cd_model_exp_00_" + dataset_name +'_seed_'+ str(sampling_params['seed']) + ".pt"
            #    torch.save(model.state_dict(), PATH)
            #    ss_params['warm_start_path'] = PATH
                

        kappa_limit = gcd_utils.kappa_coeff(F_ask.flatten().astype(bool),
                                            gcd_utils.spixels_upsampling(f,segments).flatten().astype(bool))
        

        #ax0 = plt.gca()
        ax[stype_idx].plot(label_rate,kappa[:,0], label = 'gsm')
        ax[stype_idx].plot(label_rate,kappa[:,1], label = 'gcn')
        ax[stype_idx].plot(label_rate,kappa[:,2], label = 'sgcn')
        ax[stype_idx].plot(label_rate,kappa_limit*np.ones(label_rate.shape), label = 'fundamental limit')
        ax[stype_idx].set_xlabel('label_rate')
        ax[stype_idx].set_ylabel('kappa coeff')
        if stype_idx == 0:
            ax[stype_idx].set_title(dataset_name + '\n' + stype[stype_idx])
        else:
            ax[stype_idx].set_title(stype[stype_idx])
        #ax[stype_idx].set_ylim(0,1)
        if stype_idx == 0:
            ax[stype_idx].legend()
    plt.tight_layout()

    #plt.savefig( "./figs\\"+ "fig_exp_kappa_vs_label_rate_" + dataset_name +'_'+ stype + ".png", dpi=600, bbox_inches='tight')


    import os    
    # Directory
    directory = "results_kappa_vs_labelrate_exp"  
    # Parent Directory path
    parent_dir = "./figs"  
    # Path
    path = os.path.join(parent_dir, directory)  
    if not os.path.isdir(path):    
        # Create the directory 'results_ss_exp' in parent_dir, "./figs"
        os.mkdir(os.path.join(parent_dir, directory))
    ofn = lambda i, seed : "fig_exp_kappa_vs_label_rate_" + dataset_name + '_seed_' + str(seed) + "_v_" + str(i) + ".png"
    cntr = 0
    path_to_file = os.path.join(path, ofn(cntr, sampling_params['seed']))
    gcd_utils.save_figure(path_to_file)


def exp_effect_of_theta(k_list, p, dataset_name, f_ask, F, X_mean, X,segments,stype):

    """ 
        Parameters:
        p : float, value between 0 and 1, which denotes the fraction of minority class labels to be taken from each class for training.
    
    """

    n = X_mean.shape[0] # number of nodes        
    n_pos = np.sum(F[:,0])
    n_neg = np.sum(F[:,1])
    if not n_neg + n_pos == n:
        raise ValueError('The number of positive and negative examples must add up to the number of nodes n.')
    

    # create sampling matrix that takes k labels per class uniformly at random
    #m = int(round(p*n_pos))
    nclasses = np.unique(f_ask)
    label_rate = (p*n_pos)*len(nclasses)/n

    sampling_params = {'seed': 0, 'type': stype, 'labels': F[:,0]}
    S = gcd_utils.construct_sampling_set(label_rate,n,sampling_params,matrix_form = True)
    #S = gcd_utils.construct_sampling_matrix(m,n,stype,seed,F)

    Y = S @ F

    y = np.zeros((n,1))
    y[:] = np.nan
    y[(S.T @ Y)[:,0]==1] = 1
    y[(S.T @ Y)[:,1]==1] = -1

    plt.figure()
    plt.imshow(gcd_utils.spixels_upsampling(y, segments))


    cd_methods = ['gsm', 'gcn', 'sgcn']
    #cd_methods = ['ss_gcn']
    #cd_methods = ['gsm', 'nystrom', 'sc']
    #cd_methods = ['gsm', 'sc']

    fig, ax = plt.subplots(len(k_list), len(cd_methods)+1, figsize=(10, 10), sharex=True, sharey=True)

    for k_idx in range(len(k_list)):
        # Learn graph's adj mtx using only spixels mean value. This seems to be optimal both for gsm and gcn

        W, theta = gcd_utils.construct_adj_matrix(X_mean, k = k_list[k_idx], knn_edge_cons=True, k_ = 2*k_list[k_idx])
        W = W/np.max(W[W>0])  
        #W = W/np.sum(np.sum(W)) 

        for m_idx in range(len(cd_methods)):
                if cd_methods[m_idx] == 'gsm':
                    f_hat = gcd_models.gsm_estimation(S,F,W)
                elif cd_methods[m_idx] == 'gcn':
                    n_hidden_channels = 16#X.shape[1]
                    f_hat,history,model = gcd_models.gcn_estimation(S,F,W,X,n_hidden_channels, ss_params = {'type': 'without'})
                elif cd_methods[m_idx] == 'sgcn':
                    K = 2 
                    f_hat,history,model = gcd_models.sgcn_estimation(S,F,W,X,K=K, ss_params = {'type': 'without'})
                elif cd_methods[m_idx] == 'ss_gcn':
                    n_hidden_channels = 16#X.shape[1]
                    f_hat = gcd_models.ss_gcn_estimation(S,F,W,X,n_hidden_channels)
                elif cd_methods[m_idx] == 'nystrom':
                    #f_hat = gcd_models.nystrom_estimation(W,200)
                    trials = 10
                    s = int(n/8)
                    f0 = F[:,0]
                    f_hat = gcd_models.nystrom_estimation(W,s)
                    kappa_f_hat = gcd_utils.kappa_coeff( f0.flatten().astype(bool), f_hat.flatten().astype(bool))
                    print(kappa_f_hat)

                    for i in range(10):
                        f_i = gcd_models.nystrom_estimation(W,s)
                        kappa_i = gcd_utils.kappa_coeff( f0.flatten().astype(bool), f_i.flatten().astype(bool))
                        print(kappa_i)
                        if kappa_i > kappa_f_hat:
                            f_hat = f_i
                            kappa_f_hat = kappa_i 
                elif cd_methods[m_idx] == 'sc':
                    f_hat = gcd_models.sc_estimation(W, segments, F[:,0])

                    f_hat[F[:,1].astype(bool)] = 0
                    f_hat[f_hat>0] = 1
           
                c_hat = gcd_utils.spixels_upsampling( f_hat, segments )
                kappa_c_hat = gcd_utils.kappa_coeff( f_ask.flatten().astype(bool), c_hat.flatten().astype(bool) )

                ax[k_idx, m_idx].imshow(c_hat)
                ax[k_idx, m_idx].set_xlabel('\u03BA='+str(np.round(kappa_c_hat,3)))
                #ax[m_idx, theta_idx].set_title(r'$\theta$'+'=' + str(theta_list[theta_idx]) +'('+'\u03BA='+str(np.round(kappa_c_hat,3))+')')
                if m_idx == 0:
                    ax[k_idx, m_idx].set_ylabel('$k$='  + str(k_list[k_idx]) + ', \n' + r'$\theta$'+'=' + str(np.round(theta,3)))
                if k_idx == 0:
                    ax[k_idx, m_idx].set_title(cd_methods[m_idx])

                ax[k_idx, m_idx].set_xticks([])
                ax[k_idx, m_idx].set_yticks([])

    for k_idx in range(len(k_list)):
        if k_idx == 1:
            ax[k_idx, len(cd_methods)].imshow(f_ask)
            ax[k_idx, len(cd_methods)].set_ylabel('Ground truth')
        ax[k_idx,len(cd_methods)].set_axis_off()             
      
    plt.savefig( "./figs\\results_cmap_estimates\\"+ "fig_exp_effect_of_theta_"+ dataset_name +".png", dpi=600, bbox_inches='tight')

#    plt.savefig( ".>figs\\"+ "fig_2_Exp1_CaseStudy"+str(int(case_study))+"_accuracy.png", dpi=600,
#    bbox_extra_artists=(lgd,),
#    bbox_inches='tight')

def exp_effect_of_random_sampling(k_list, p, dataset_name, f_ask,F,X_mean, X,segments,stype):

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


    #cd_methods = ['gsm', 'gcn', 'nystrom']
    cd_methods = ['nystrom']

    #fig, ax = plt.subplots(len(cd_methods)+1, len(k_list), figsize=(10, 10), sharex=True, sharey=True)

    

    for k_idx in range(len(k_list)):
        # Learn graph's adj mtx using only spixels mean value. This seems to be optimal both for gsm and gcn

        W, theta = gcd_utils.construct_adj_matrix(X_mean, k = k_list[k_idx], knn_edge_cons=True, k_ = 2*k_list[k_idx])
        W = W/np.max(W[W>0])   
        print(dataset_name)
        s = int(n/8)
        #s = 200
        print('number of nodes:', str(s))
        for i in range(10):            
            f_hat = gcd_models.nystrom_estimation(W,s)            
            c_hat = gcd_utils.spixels_upsampling( f_hat, segments )
            kappa_c_hat = gcd_utils.kappa_coeff( f_ask.flatten().astype(bool), c_hat.flatten().astype(bool) )
            print(kappa_c_hat)

    
      
    #plt.savefig( "./figs\\"+ "fig_exp_effect_of_theta_"+ dataset_name +"112.png", dpi=600, bbox_inches='tight')

#    plt.savefig( ".>figs\\"+ "fig_2_Exp1_CaseStudy"+str(int(case_study))+"_accuracy.png", dpi=600,
#    bbox_extra_artists=(lgd,),
#    bbox_inches='tight')

def exp_accuracy_vs_ss_level(k, p, nclasses_ss, dataset_name, f_ask, F, X_mean, X,segments,stype):

    # 
    n = X_mean.shape[0]

    # 1. Learn graph from X_mean
    W, theta = gcd_utils.construct_adj_matrix(X_mean, k = k, knn_edge_cons=True, k_ = 2*k)
    W = W/np.max(W[W>0])  

    # 2. Construct sampling matrix to generate known labels from F
    y = 1 - np.argmax(F,axis=1)
    class_labels, class_counts = np.unique(y,return_counts=True)
    label_rate = (p*np.min(class_counts))*len(class_labels)/n
    seed = 0
    stype='random-class-dependent'
    sampling_params = {'type': stype, 'seed' : seed, 'labels': y}
    S = gcd_utils.construct_sampling_set(label_rate, n, sampling_params, matrix_form = True)

    hidden_chs = 16
    theta0, sigma = gcd_models.ss_gcn_estimation(W,nclasses_ss,X,hidden_chs)
    sigma = 1

    #nclasses_ss = 5
    
    rho = [0,0.01,0.1,1,10,100,1000,10000,np.inf]
    delta = np.zeros((12,1))
    delta[0]  = np.inf 
    delta[1:11] = np.reshape(np.logspace(2, -5, num=10),(10,1))
    delta[11]  = 0     
    rho = 4.47*sigma/delta
    rho[0] = 0
    rho[-1] = np.inf
    kappa = np.zeros((len(rho),1))
    train_loss = np.zeros((len(rho),1))
    ss_loss = np.zeros((len(rho),1))

    # Define self supervised learning params
    ss_params = {'type': 'without',
                 'theta0': theta0}

    for rho_idx in range(len(rho)):
          
        ss_params['rho'] = rho[rho_idx][0]
        if ss_params['rho'] == 0:
            ss_params['type'] = 'without' 
        elif ss_params['rho'] == np.inf:
            ss_params['type'] = 'tf'
        else:
            ss_params['type'] = 'reg'   


        f_hat, history, model = gcd_models.gcn_estimation(S, F, W, X, hidden_chs, 
                                                   ss_params = ss_params)

        if rho_idx == 0:
            # Specify a path to save to
            PATH = "./pretrained_models\\" + "gcn_cd_model_" + dataset_name + ".pt"
            torch.save(model.state_dict(), PATH)
            ss_params['warm_start_path'] = PATH


        kappa[rho_idx] = gcd_utils.kappa_coeff( y.flatten().astype(bool), f_hat.flatten().astype(bool))
        train_loss[rho_idx] = history['loss'][-1]
        ss_loss[rho_idx] = history['ss_loss'][-1]  

    fig, ax = plt.subplots(3,1,sharex=True)#figsize=(10, 10), sharex=True, sharey=True)
    ax[0].semilogx(np.asarray(rho[0:len(rho)-1]).reshape(len(rho)-1,1),kappa[0:len(rho)-1])
    #ax[0].semilogx([rho[-2]],[kappa[-1]],marker="o", markeredgecolor="red")
    #ax[0].set_xlabel('Self-supervision level (rho)')
    ax[0].set_ylabel('kappa coefficient')
    ax[0].set_title(dataset_name)

    ax[1].semilogx(np.asarray(rho[0:len(rho)-1]).reshape(len(rho)-1,1),train_loss[0:len(rho)-1])
    #ax[1].semilogx([rho[-2]],[train_loss[-1]],marker="o", markeredgecolor="red")
    #ax[1].set_xlabel('Self-supervision level (rho)')
    ax[1].set_ylabel('Train loss')
    ax[2].semilogx(np.asarray(rho[0:len(rho)-1]).reshape(len(rho)-1,1),ss_loss[0:len(rho)-1])
    #ax[2].semilogx(np.asarray(rho[0:len(rho)-1]).reshape(len(rho)-1,1),delta[0:len(rho)-1]/(hidden_chs*n))
    #ax[2].semilogx([rho[-2]],[train_loss[-1]],marker="o", markeredgecolor="red")
    ax[2].set_xlabel('Self-supervision level (rho)')
    ax[2].set_ylabel('ss loss')
    #ax[0].set_title('Dataset: ' + dataset_name)

    #import os.path
    #from os import path
    ofn = lambda i, k : "fig_exp_kappa_vs_ss_level_"+ dataset_name +"_clusters_"+ str(k) + "_v_" + str(i) + ".png"
    cntr = 0

    import os    
    # Directory
    directory = "results_ss_exp"  
    # Parent Directory path
    parent_dir = "./figs"  
    # Path
    path = os.path.join(parent_dir, directory)  
    if not os.path.isdir(path):    
        # Create the directory 'results_ss_exp' in parent_dir, "./figs"
        os.mkdir(os.path.join(parent_dir, directory))
    path_to_file = os.path.join(path, ofn(cntr, nclasses_ss)) #"./figs\\results_ss_exp\\" + ofn(cntr, nclasses_ss)
    gcd_utils.save_figure(path_to_file)

    
def exp_accuracy_vs_reg_level(k, p, dataset_name, f_ask, F, X_mean, X,segments,stype):

    # 
    n = X_mean.shape[0]

    # 1. Learn graph from X_mean
    W, theta = gcd_utils.construct_adj_matrix(X_mean, k = k, knn_edge_cons=True, k_ = 2*k)
    W = W/np.max(W[W>0])  

    # 2. Construct sampling matrix to generate known labels from F
    y = 1 - np.argmax(F,axis=1)
    class_labels, class_counts = np.unique(y,return_counts=True)
    label_rate = (p*np.min(class_counts))*len(class_labels)/n
    seed = 0
    stype='random-class-dependent'
    sampling_params = {'type': stype, 'seed' : seed, 'labels': y}
    S = gcd_utils.construct_sampling_set(label_rate, n, sampling_params, matrix_form = True)

    hidden_chs = 16
    #theta0, sigma = gcd_models.ss_gcn_estimation(W,nclasses_ss,X,hidden_chs)
    sigma = 1

    #nclasses_ss = 5
    
    rho = [0,0.01,0.1,1,10,100,1000,10000,np.inf]
    #delta = np.zeros((12,1))
    #delta[0]  = np.inf 
    #delta[1:11] = np.reshape(np.logspace(2, -5, num=10),(10,1))
    #delta[11]  = 0     
    #rho = 4.47*sigma/delta
    #rho[0] = 0
    #rho[-1] = np.inf
    kappa = np.zeros((len(rho),1))
    train_loss = np.zeros((len(rho),1))
    ss_loss = np.zeros((len(rho),1))

    # Define self supervised learning params
    ss_params = {'type': 'without'}

    for rho_idx in range(len(rho)):
          
        ss_params['rho'] = rho[rho_idx]
        if ss_params['rho'] == 0:
            ss_params['type'] = 'without' 
        elif ss_params['rho'] == np.inf:
            ss_params['type'] = 'tf'
        else:
            ss_params['type'] = 'reg'   


        f_hat, history, model = gcd_models.gsmgcn_estimation(S, F, W, X, hidden_chs, 
                                                   ss_params = ss_params)

        if rho_idx >= 0:
            # Specify a path to save to
            PATH = "./pretrained_models\\" + "gcn_cd_model_" + dataset_name + ".pt"
            torch.save(model.state_dict(), PATH)
            ss_params['warm_start_path'] = PATH


        kappa[rho_idx] = gcd_utils.kappa_coeff( y.flatten().astype(bool), f_hat.flatten().astype(bool))
        train_loss[rho_idx] = history['loss'][-1]
        ss_loss[rho_idx] = history['ss_loss'][-1]  

    fig, ax = plt.subplots(3,1,sharex=True)#figsize=(10, 10), sharex=True, sharey=True)
    ax[0].semilogx(np.asarray(rho[0:len(rho)-1]).reshape(len(rho)-1,1),kappa[0:len(rho)-1])
    #ax[0].semilogx([rho[-2]],[kappa[-1]],marker="o", markeredgecolor="red")
    #ax[0].set_xlabel('Self-supervision level (rho)')
    ax[0].set_ylabel('kappa coefficient')
    ax[0].set_title(dataset_name)

    ax[1].semilogx(np.asarray(rho[0:len(rho)-1]).reshape(len(rho)-1,1),train_loss[0:len(rho)-1])
    #ax[1].semilogx([rho[-2]],[train_loss[-1]],marker="o", markeredgecolor="red")
    #ax[1].set_xlabel('Self-supervision level (rho)')
    ax[1].set_ylabel('Train loss')
    ax[2].semilogx(np.asarray(rho[0:len(rho)-1]).reshape(len(rho)-1,1),ss_loss[0:len(rho)-1])
    #ax[2].semilogx(np.asarray(rho[0:len(rho)-1]).reshape(len(rho)-1,1),delta[0:len(rho)-1]/(hidden_chs*n))
    #ax[2].semilogx([rho[-2]],[train_loss[-1]],marker="o", markeredgecolor="red")
    ax[2].set_xlabel('Self-supervision level (rho)')
    ax[2].set_ylabel('ss loss')
    #ax[0].set_title('Dataset: ' + dataset_name)

    #import os.path
    #from os import path
    ofn = lambda i, k : "fig_exp_kappa_vs_ss_level_"+ dataset_name +"_clusters_"+ str(k) + "_v_" + str(i) + ".png"
    cntr = 0

    import os    
    # Directory
    directory = "results_ss_exp"  
    # Parent Directory path
    parent_dir = "./figs"  
    # Path
    path = os.path.join(parent_dir, directory)  
    if not os.path.isdir(path):    
        # Create the directory 'results_ss_exp' in parent_dir, "./figs"
        os.mkdir(os.path.join(parent_dir, directory))
    path_to_file = os.path.join(path, ofn(cntr, nclasses_ss)) #"./figs\\results_ss_exp\\" + ofn(cntr, nclasses_ss)
    gcd_utils.save_figure(path_to_file)


    


        




