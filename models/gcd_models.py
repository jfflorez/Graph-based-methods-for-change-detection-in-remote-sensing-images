from json import tool
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy as sp
import scipy.sparse.linalg as sp_linalg

import utils.gcd_utils as gcd_utils
from scipy.sparse import coo_matrix, hstack, vstack, find
from scipy.sparse.linalg import spsolve
from scipy.stats import skew

from sklearn_extra.cluster import KMedoids
from sklearn.metrics import accuracy_score

import torch
import torch.nn.functional as functional
#import sys  
# adding Folder_2 to the system path
#sys.path.insert(0, 'C:\\Users\\juanf\\.conda\\envs\\gcnn_env\\Lib\site-packages\\torch_sparse\\_convert_cuda.pyd')
from torch_geometric.nn import GCNConv, SGConv

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import davies_bouldin_score

def gsm_estimation(S,F,W):
        n = W.shape[0]
        I = sp.sparse.eye(n)
        #D = scipy.sparse.spdiags(np.sum(W,1).squeeze(),0, W.shape[0], W.shape[1])
        D_sq_inv = sp.sparse.spdiags(np.power(np.sum(W,1),-0.5).squeeze(),0, W.shape[0], W.shape[1])
        L = I - D_sq_inv @ (W @ D_sq_inv)

        Y = S@F
        zeros = coo_matrix((S.shape[0], S.T.shape[1]), dtype=np.float64)
        A = vstack( [hstack([L,S.T]), hstack([S,zeros.tocsr()])] )
        zeros_b = coo_matrix((S.T.shape[0],1), dtype=np.float64)
        F_hat = np.zeros((n,2)) 
        b_0 = vstack([zeros_b,Y[:,0].reshape((Y[:,0].shape[0],1))])
        F_hat[:,0] = spsolve(A, b_0)[0:n]
        b_1 = vstack([zeros_b,Y[:,1].reshape((Y[:,1].shape[0],1))])
        F_hat[:,1] = spsolve(A, b_1)[0:n]
        f_hat = 1-np.argmax(F_hat,axis=1)
        return f_hat

def nystrom_estimation(W,k):

    n = W.shape[0]

    D_sq_inv = sp.sparse.spdiags(np.power(np.sum(W,1),-0.5).squeeze(),0, W.shape[0], W.shape[1])
    W = D_sq_inv @ (W @ D_sq_inv)

    idx = np.arange(0,n)
    #maxIt = 1000
    #bn_sampling_pattern = gcd_utils.blue_noise_sampling(W,k,maxIt)
    #rnd_idx = np.argsort(-100*bn_sampling_pattern + np.random.rand(n,1),axis=0).squeeze().tolist()
    #np.random.seed(10)
    rnd_idx = np.argsort(np.random.rand(n,1),axis=0).squeeze().tolist()

   # p = np.zeros((n,1)).squeeze()
   # p[rnd_idx[0:k]] = 1

    #idx_p0 = idx[rnd_idx[0:k]]
    #idx[rnd_idx[0:k]] = -10
    #idx_p1 = idx[idx != -10]
    #idx_p = np.concatenate((idx_p0,idx_p1))
    idx_p = idx[rnd_idx]    
    P = coo_matrix((np.ones((n,1)).squeeze(),(idx_p,np.arange(0,n))), shape=(n,n)).tocsr()
    W = P.T @ (W @ P)

    ##
    S_kxk, U_kxk = sp.linalg.eigh(W[0:k,0:k].toarray())
    S_kxk_inv = 1/S_kxk
    S_kxk_inv[S_kxk==0] = 0
    #print('Aprox. error:', np.linalg.norm((W[k:n,k:n] - W[k:n,0:k] @ (U_kxk.T @ (np.diag(S_kxk_inv) @ U_kxk)) @ W[0:k,k:n]),ord='fro'))
    ##


    #filter = S_kxk <= np.mean(S_kxk) + np.std(S_kxk)/2
    #filter = np.logical_and(filter,S_kxk >= np.mean(S_kxk) - np.std(S_kxk)/2) 
    #S_kxk_inv[filter] = 0

    #  W[k:n,0:k] <=>  W[0:k,k:n].T    
    U = (P @ vstack((U_kxk, W[k:n,0:k] @ ( U_kxk @ np.diag(S_kxk_inv)))))
    U = U @ np.diag(np.abs(S_kxk))
    #print('Aprox. error:', np.linalg.norm((W[k:n,k:n] - W[k:n,0:k] @ (U_kxk.T @ (np.diag(S_kxk_inv) @ U_kxk)) @ W[0:k,k:n]),ord='fro'))

    kmeans = KMeans(n_clusters=2, random_state=0, tol=1e-8).fit(U)
    f_hat = kmeans.labels_

    #kmedoids = KMedoids(n_clusters=2, random_state=0, max_iter=1000).fit(U)
    #f_hat = kmedoids.labels_
    

    #print('Cost function value:', kmeans.inertia_)

    #S = sp.sparse.eye(n).tocsr()[rnd_idx[0:k],:]

    #F = np.empty((n,2))
    #F[:] = np.nan
    #F[rnd_idx[0:k],0] = f0
    #F[rnd_idx[0:k],1] = 1-f0

    #f_hat = gsm_estimation(S,F,W)

    return f_hat


def sc_estimation(W, segments, f_ask):

        n = W.shape[0]

        D_sq_inv = sp.sparse.spdiags(np.power(np.sum(W,1),-0.5).squeeze(),0, W.shape[0], W.shape[1])
        I = sp.sparse.spdiags(np.ones((n,1)).squeeze(),0, n, n)

        S, U = sp.linalg.eigh((I - D_sq_inv @ (W @ D_sq_inv)).toarray())
        d = int(n/8)
        kappa_score = np.zeros((d,1))
        kappa_f_hat =  -100
        for j in range(9,10):
            #kmeans = KMeans(n_clusters=2, random_state=0, tol=1e-8).fit(U[:,0:j])
            kmeans = KMeans(n_clusters=j, random_state=0, tol=1e-8).fit(U[:,0:d])
            f_j = kmeans.labels_
            #kappa_score[j-1] = gcd_utils.kappa_coeff(f_ask.flatten().astype(bool), f_j.flatten().astype(bool) )
            #kappa_score[j-1] = adjusted_rand_score(f_ask.flatten().astype(int),f_j.flatten().astype(int)) 
            kappa_score[j-1] = davies_bouldin_score(U[:,0:d],f_j)
            if kappa_score[j-1] > kappa_f_hat:
                f_hat = f_j
                kappa_f_hat = kappa_score[j-1] 
            

        #j_best = np.argmax(kappa_score)
        #kmeans = KMeans(n_clusters=2, random_state=0, tol=1e-8).fit(U[:,0:j_best])
        #f_hat = kmeans.labels_

        return f_hat

def sc_labeler(W, nclusters):

        n = W.shape[0]

        D_sq_inv = sp.sparse.spdiags(np.power(np.sum(W,1),-0.5).squeeze(),0, W.shape[0], W.shape[1])
        I = sp.sparse.spdiags(np.ones((n,1)).squeeze(),0, n, n)

        S, U = sp.linalg.eigh((I - D_sq_inv @ (W @ D_sq_inv)).toarray())
        d = 2*16 
        #J = np.zeros((20,1))
        #for i in range(2,len(J)):
        #    kmeans = KMeans(n_clusters = i, random_state=0, tol=1e-8).fit(U[:,0:d]/np.reshape(np.sqrt(np.sum(U[:,0:d]**2,axis=1)),(n,1)))
        #    J[i-2] = kmeans.inertia_
        #kmeans = KMeans(n_clusters = nclusters, random_state=0, tol=1e-8).fit(U[:,0:d])        
        kmeans = KMeans(n_clusters = nclusters, random_state=0, tol=1e-8).fit(U[:,0:d]/np.reshape(np.sqrt(np.sum(U[:,0:d]**2,axis=1)),(n,1)))
        y =  kmeans.labels_
        class_labels, class_counts = np.unique(y, return_counts=True)

        c_largest = np.argmax(class_counts)
        # Compute relative distance to largest cluster
        dist_2_cmax = np.sqrt(np.sum((kmeans.cluster_centers_ - kmeans.cluster_centers_[c_largest,:])**2,1))
        # Compute permutation that sorts elements of dist_2_max in ascending order
        idx = np.argsort(dist_2_cmax)
        # Pick 
        c0 = idx[-1]
        y[np.logical_not(np.logical_or(y==c_largest,y==c0 ))] = nclusters+1  

        nw_class_labels =  np.sort([c0,c_largest,nclusters+1])
        for i in range(0,len(nw_class_labels)):
            y[y==nw_class_labels[i]] = i
        class_labels, class_counts = np.unique(y, return_counts=True)
        
        # Construct and return one-hot encoded labels
        # Y = np.zeros((n,nclusters))
        # Y[:,y] = 1         

        return y, class_labels, class_counts

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,add_self_loops=True,bias=False)
        self.conv2 = GCNConv(hidden_channels, out_channels,add_self_loops=True,bias=False)
        self.dropout_rate = 0

    def set_dropout_rate(self,dropout_rate):
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, edge_weight):
        #x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index, edge_weight)
        x = functional.dropout(x, p = self.dropout_rate, training=self.training)
        #x = self.conv1(x, edge_index)
        x = functional.relu(x)
        #x = functional.dropout(x, p = 0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        #x = self.conv2(x, edge_index)
        return functional.log_softmax(x, dim=1)

class SGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.conv = SGConv(in_channels, out_channels, K, add_self_loops=True,bias=False)
        #self.conv2 = GCNConv(hidden_channels, out_channels,add_self_loops=True,bias=False)
        #self.dropout_rate = 0

    def forward(self, x, edge_index, edge_weight):
        #x, edge_index = data.x, data.edge_index
        x = self.conv(x, edge_index, edge_weight)
        return functional.log_softmax(x, dim=1)

def ss_gcn_estimation(W,nclasses_ss,X,hidden_chs):
    """
    Parameters:
    S : sparse k by n sampling matrix, allowing the selection of training examples.
    F : original one hot encoded label matrix of shape (n, num_of_classes)
    W : graph adjancecy matrix of shape (n,n)¨
    X : feature vector matrix of shape (n,d)
    """
    # Compute number of nodes and number of features
    n, num_of_features = X.shape

    X = torch.tensor(X,dtype =torch.float32)
    
    # 1. Construct self-supervised model from self-supervised labels:

    # 1.1 Generate pseudo labels using spectral clustering 
    y_ss, class_labels, class_counts = sc_labeler(W, nclasses_ss) #.astype(np.int64)
    y_ss = y_ss.astype(np.int64)
    nclasses_ss = len(class_labels)
    
    # 1.2 Initialize GCN model and optimizer
    model_ss = GCN(num_of_features, hidden_chs, nclasses_ss)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer = torch.optim.Adamax(model_ss.parameters(), lr = 0.01, weight_decay = 10e-4)#5e-4)    
    
    # Convert adjacency matrix to pytorch format 
    output = find(W)
    edge_indices =torch.tensor(np.vstack((output[0],output[1])),dtype =torch.int64)
    edge_weights = torch.tensor(output[2],dtype = torch.float32)

    # 2. Train self-supervised model 
    seed = 0
    p = 0.5
    label_rate = 0.5 #np.median(class_counts)*nclasses_ss/n
    D_sq_inv = sp.sparse.spdiags(np.power(np.sum(W,1),-0.5).squeeze(),0, W.shape[0], W.shape[1])
    sampling_params = {'type': 'blue-noise', 'seed' : seed, 'W': D_sq_inv @ (W @ D_sq_inv)}
    idx_ss = gcd_utils.construct_sampling_set(label_rate, n, sampling_params, matrix_form = False)
    y_ss = torch.from_numpy(y_ss)

    state_dict0 = model_ss.state_dict()
    print('self-supervised task begins: \n')
    #dropout_rate = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #dropout_rate = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    dropout_rate = [0.1]
    #dropout_rate = [0.1,0.9]
    N = len(dropout_rate)
    loss_values = np.zeros((N,1))
    for i in range(len(dropout_rate)):

        model_ss.load_state_dict(state_dict0,strict=True)

        model_ss.train()
        model_ss.set_dropout_rate(dropout_rate[i])

        for epoch in range(nclasses_ss*80):
            optimizer.zero_grad()
            out = model_ss(X,edge_indices,edge_weights)
            loss = functional.nll_loss(out[idx_ss,:], y_ss[idx_ss], weight= torch.Tensor(1/class_counts))
            print(epoch, loss.data)
            loss.backward()
            optimizer.step()
        loss_values[i] = loss.detach().numpy()

        model_ss.eval()
        f_hat_ss = model_ss(X,edge_indices,edge_weights).argmax(dim=1)        
        test_samples = np.ones((n,1)).astype(bool)
        test_samples[idx_ss] = False
        print('Train accuracy:', accuracy_score(y_ss[idx_ss], f_hat_ss[idx_ss]))
        print ('Test accuracy:', accuracy_score(y_ss[test_samples.squeeze()], f_hat_ss[test_samples.squeeze()]))

        # Return parameters associated with feature extraction layer

        # 3. Store layer filters of ss model to regularize the downstream task
        #layer_filters = {}
        for name, theta in model_ss.named_parameters():
            #print(((theta.flatten()**2).sum()).sqrt().detach())
            if 'conv1' in name:
                theta0 = torch.tensor(theta.detach())
                #if i > 1:
                #    theta0 = theta0 + torch.tensor(theta.detach())
                #else:
                #    theta0 = torch.tensor(theta.detach())
                #theta0 = theta0/((theta0.flatten()**2).sum().sqrt())
                #layer_filters[name] = torch.tensor(theta/((theta.flatten()**2).sum()).sqrt().detach())
    return theta0, np.std(loss_values)    


def ss_gcn_estimation_v02(W,y_ss,X,hidden_chs):
        """
        Parameters:
        S : sparse k by n sampling matrix, allowing the selection of training examples.
        F : original one hot encoded label matrix of shape (n, num_of_classes)
        W : graph adjancecy matrix of shape (n,n)¨
        X : feature vector matrix of shape (n,d)
        """

        # Compute number of nodes and number of features
        n, num_of_features = X.shape

        X = torch.tensor(X,dtype =torch.float32)
        
        # 1. Construct self-supervised model from self-supervised labels:

        # 1.1 Generate pseudo labels using spectral clustering 
        #y_ss = sc_labeler(W, nclasses_ss).astype(np.int64)
        class_labels, class_counts = np.unique(y_ss,return_counts=True)
        nclasses_ss = len(class_labels)
        # 1.2 Initialize GCN model and optimizer
        model_ss = GCN(num_of_features,hidden_chs, nclasses_ss)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        optimizer = torch.optim.Adamax(model_ss.parameters(), lr = 0.01, weight_decay = 10e-4)#5e-4)        
        
        # Convert adjacency matrix to pytorch format 
        output = find(W)
        edge_indices =torch.tensor(np.vstack((output[0],output[1])),dtype =torch.int64)
        edge_weights = torch.tensor(output[2],dtype = torch.float32)

        # 2. Train self-supervised model 
        seed = 0
        p = 0.5
        label_rate = 0.75 #np.min(class_counts)*nclasses_ss/n
        #stype = 'random-class-dependent'
        #idx_ss = gcd_utils.construct_sampling_set(label_rate,n,stype,seed,y_ss)
        D_sq_inv = sp.sparse.spdiags(np.power(np.sum(W,1),-0.5).squeeze(),0, W.shape[0], W.shape[1])
        sampling_params = {'type': 'blue-noise', 'seed' : seed, 'W': D_sq_inv @ (W @ D_sq_inv)}
        idx_ss = gcd_utils.construct_sampling_set(label_rate, n, sampling_params, matrix_form = False)
        y_ss = torch.from_numpy(y_ss)

        print('self-supervised task begins: \n')
        model_ss.train()
        for epoch in range(nclasses_ss*80):
            optimizer.zero_grad()
            out = model_ss(X,edge_indices,edge_weights)
            loss = functional.nll_loss(out[idx_ss,:], y_ss[idx_ss])
            print(loss.data)
            loss.backward()
            optimizer.step()
        model_ss.eval()
        
        f_hat_ss = model_ss(X,edge_indices,edge_weights).argmax(dim=1)

        from sklearn.metrics import accuracy_score
        test_samples = np.ones((n,1)).astype(bool)
        test_samples[idx_ss] = False
        print('Train accuracy:', accuracy_score(y_ss[idx_ss], f_hat_ss[idx_ss]))
        print ('Test accuracy:', accuracy_score(y_ss[test_samples.squeeze()], f_hat_ss[test_samples.squeeze()]))

        # Return parameters associated with feature extraction layer

        # 3. Store layer filters of ss model to regularize the downstream task
        #layer_filters = {}
        for name, theta in model_ss.named_parameters():
            #print(((theta.flatten()**2).sum()).sqrt().detach())
            if 'conv1' in name:
                theta0 = torch.tensor(theta.detach())
                #theta0 = theta0/((theta0.flatten()**2).sum().sqrt())
            #layer_filters[name] = torch.tensor(theta/((theta.flatten()**2).sum()).sqrt().detach())
        return theta0

#from torch_geometric.nn import GCNConvCould not find module 'C:\Users\juanf\.conda\envs\gcnn_env\Lib\site-packages\torch_sparse\_convert_cuda.pyd' (or one of its dependencies). Try using the full path with constructor syntax.
def gcn_estimation(S, F, W, X, hidden_chs, ss_params = {'ss_required': False}):
    """
    Parameters:
    S : sparse k by n sampling matrix, allowing the selection of training examples.
    F : original one hot encoded label matrix of shape (n, num_of_classes)
    W : graph adjancecy matrix of shape (n,n)¨
    X : feature vector matrix of shape (n,d)
    """
    n = W.shape[0] # number of nodes
    n_classes = F.shape[1]

    X = torch.tensor(X,dtype =torch.float32)
    num_of_features = X.shape[1]

    model = GCN(num_of_features,hidden_chs,n_classes)

    if 'warm_start_path' in ss_params.keys():
        model.load_state_dict(torch.load(ss_params['warm_start_path']), strict=True)

    if ss_params['type'] == 'reg':
        theta0 = ss_params['theta0'] # ss_gcn_estimation(W,nclasses_ss,X,hidden_chs)
        rho = ss_params['rho']
        for param_name, param in model.named_parameters():
            if 'conv1' in param_name:
                theta = param
    elif ss_params['type'] == 'tf':
        for param_name, param in model.named_parameters():
            if 'conv1' in param_name:
                param.data = ss_params['theta0']
                param.requires_grad = False

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    #optimizer = torch.optim.Adamax(model.parameters(), lr=0.01) #, weight_decay=5e-4)
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.01, weight_decay=10e-4)

    # adjacency matrix conversion to pytorch format 
    output = find(W)
    edge_indices =torch.tensor(np.vstack((output[0],output[1])),dtype =torch.int64)
    edge_weights = torch.tensor(output[2],dtype = torch.float32)

    # training labels
    Y = S@F
    y_train = torch.from_numpy(1-np.argmax(Y,axis=1))
    idx_train = (S@np.arange(n).reshape((n,1)).squeeze()).astype(int)   

    maxit = int(300)
    #tol = 1e-2
    history = {'loss': np.zeros((maxit,1)),
               'ss_loss': np.zeros((maxit,1))}
    model.train()
    #model.set_dropout_rate(0.1)
    dropout_rate = 0.0
    for epoch in range(1,maxit):
        
        optimizer.zero_grad()
        out = model(X,edge_indices,edge_weights)

        if ss_params['type'] == 'reg':
            #loss_ss = -(theta*theta0).sum() 
            loss_ss = functional.dropout(((theta-theta0)**2).sum()/((1-dropout_rate)*n*hidden_chs),p=dropout_rate) #functional.mse_loss(theta,theta0)
            #loss_ss = ((theta-theta0)**2).sum()/(n*hidden_chs) #functional.mse_loss(theta,theta0)
            loss = functional.nll_loss(out[idx_train,:], y_train) + rho*loss_ss
            history['loss'][epoch] = loss.data - rho*loss_ss.data
            history['ss_loss'][epoch] = loss_ss.data
        else:
            loss = functional.nll_loss(out[idx_train,:], y_train)
            #loss_ss = torch.Tensor(0)
            history['loss'][epoch] = loss.data

        print('Epoch:', epoch,', loss:',loss.data)
        loss.backward()
        optimizer.step()       
    model.eval()

    f_hat = model(X,edge_indices,edge_weights).argmax(dim=1)

    return f_hat.numpy(), history, model

def sgcn_estimation(S, F, W, X, K, ss_params = {'type': 'without'}):
    """
    Parameters:
    S : sparse k by n sampling matrix, allowing the selection of training examples.
    F : original one hot encoded label matrix of shape (n, num_of_classes)
    W : graph adjancecy matrix of shape (n,n)¨
    X : feature vector matrix of shape (n,d)
    """
    n = W.shape[0] # number of nodes
    n_classes = F.shape[1]

    X = torch.tensor(X,dtype =torch.float32)
    num_of_features = X.shape[1]

    model = SGCN(num_of_features,n_classes,K)

    if 'warm_start_path' in ss_params.keys():
        model.load_state_dict(torch.load(ss_params['warm_start_path']), strict=True)

    if ss_params['type'] == 'reg':
        theta0 = ss_params['theta0'] # ss_gcn_estimation(W,nclasses_ss,X,hidden_chs)
        rho = ss_params['rho']
        for param_name, param in model.named_parameters():
            if 'conv1' in param_name:
                theta = param
    elif ss_params['type'] == 'tf':
        for param_name, param in model.named_parameters():
            if 'conv1' in param_name:
                param.data = ss_params['theta0']
                param.requires_grad = False

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    #optimizer = torch.optim.Adamax(model.parameters(), lr=0.01) #, weight_decay=5e-4)
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.01, weight_decay=10e-4)

    # adjacency matrix conversion to pytorch format 
    output = find(W)
    edge_indices =torch.tensor(np.vstack((output[0],output[1])),dtype =torch.int64)
    edge_weights = torch.tensor(output[2],dtype = torch.float32)

    # training labels
    Y = S@F
    y_train = torch.from_numpy(1-np.argmax(Y,axis=1))
    idx_train = (S@np.arange(n).reshape((n,1)).squeeze()).astype(int)   

    maxit = int(300)
    #tol = 1e-2
    history = {'loss': np.zeros((maxit,1))}
    model.train()
    #model.set_dropout_rate(0.1)
    dropout_rate = 0.0
    for epoch in range(1,maxit):
        
        optimizer.zero_grad()
        out = model(X,edge_indices,edge_weights)
        loss = functional.nll_loss(out[idx_train,:], y_train)
            #loss_ss = torch.Tensor(0)
        history['loss'][epoch] = loss.data

        print('Epoch:', epoch,', loss:',loss.data)
        loss.backward()
        optimizer.step()       
    model.eval()

    f_hat = model(X,edge_indices,edge_weights).argmax(dim=1)

    return f_hat.numpy(), history, model

def gsmgcn_estimation(S, F, W, X, hidden_chs, ss_params = {'ss_required': False}):
    """
    Parameters:
    S : sparse k by n sampling matrix, allowing the selection of training examples.
    F : original one hot encoded label matrix of shape (n, num_of_classes)
    W : graph adjancecy matrix of shape (n,n)¨
    X : feature vector matrix of shape (n,d)
    """
    n = W.shape[0] # number of nodes
    n_classes = F.shape[1]

    #D_sq_inv = sp.sparse.spdiags(np.power(np.sum(W,1),-0.5).squeeze(),0, W.shape[0], W.shape[1])
    #L_G = sp.sparse.spdiags(np.ones((n,1)).squeeze(),0, n, n) - W

    X = torch.tensor(X,dtype =torch.float32)
    num_of_features = X.shape[1]

    model = GCN(num_of_features,hidden_chs,n_classes)

    if 'warm_start_path' in ss_params.keys():
        model.load_state_dict(torch.load(ss_params['warm_start_path']), strict=True)

    if ss_params['type'] == 'reg':
        #theta0 = ss_params['theta0'] # ss_gcn_estimation(W,nclasses_ss,X,hidden_chs)
        rho = ss_params['rho']
        #for param_name, param in model.named_parameters():
        #    if 'conv1' in param_name:
        #        theta = param
    elif ss_params['type'] == 'tf':
        for param_name, param in model.named_parameters():
            if 'conv1' in param_name:
                param.data = ss_params['theta0']
                param.requires_grad = False

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    #optimizer = torch.optim.Adamax(model.parameters(), lr=0.01) #, weight_decay=5e-4)
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.01, weight_decay=10e-4)

    # adjacency matrix conversion to pytorch format 
    output = find(W)
    edge_indices =torch.tensor(np.vstack((output[0],output[1])),dtype =torch.int64)
    edge_weights = torch.tensor(output[2],dtype = torch.float32)



    # training labels
    Y = S@F
    y_train = torch.from_numpy(1-np.argmax(Y,axis=1))
    idx_train = (S@np.arange(n).reshape((n,1)).squeeze()).astype(int)   

    maxit = int(300)
    #tol = 1e-2
    history = {'loss': np.zeros((maxit,1)),
               'ss_loss': np.zeros((maxit,1))}
    model.train()

    u = torch.from_numpy(np.random.randn(len(y_train),1).astype(np.float32))
    u.requires_grad = True

    z = np.zeros((n,1),dtype=np.float32)
    z[idx_train] = np.reshape(y_train.data.numpy(),(len(y_train),1))
    z = torch.from_numpy(z)
    z.requires_grad = True

    w = torch.sparse_coo_tensor(edge_indices, edge_weights, (n, n))
    d = torch.sparse.sum(w,1).to_dense().reshape((n,1))

    #model.set_dropout_rate(0.1)
    dropout_rate = 0.1
    for epoch in range(1,maxit):
        
        optimizer.zero_grad()
        out = model(X,edge_indices,edge_weights)

        if ss_params['type'] == 'reg':
            #loss_ss = -(theta*theta0).sum() 
            loss_sm = (z*(z-torch.matmul(w,z/d.sqrt())/d.sqrt())).sum()
            loss_lgm = (u*(z[idx_train]-y_train)).sum()
            #loss_ss = functional.dropout(((theta-theta0)**2).sum()/((1-dropout_rate)*n*hidden_chs),p=dropout_rate) #functional.mse_loss(theta,theta0)
            #loss_ss = ((theta-theta0)**2).sum()/(n*hidden_chs) #functional.mse_loss(theta,theta0)
            loss = functional.nll_loss(out[idx_train,:], z[idx_train].squeeze()) + rho*loss_sm + loss_lgm
            history['loss'][epoch] = loss.data - rho*loss_sm.data - loss_lgm
            history['sm_loss'][epoch] = loss_sm.data
        else:
            loss = functional.nll_loss(out[idx_train,:], y_train)
            #loss_ss = torch.Tensor(0)
            history['loss'][epoch] = loss.data

        print('Epoch:', epoch,', loss:',loss.data)
        loss.backward()
        optimizer.step()       
    model.eval()

    f_hat = model(X,edge_indices,edge_weights).argmax(dim=1)

    return f_hat.numpy(), history, model