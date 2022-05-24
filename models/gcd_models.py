
from cProfile import label
from matplotlib.pyplot import figure
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy as sp

import utils.gcd_utils as gcd_utils
from scipy.sparse import coo_matrix, hstack, vstack, find
from scipy.sparse.linalg import spsolve

import torch
import torch.nn.functional as functional
#import sys  
# adding Folder_2 to the system path
#sys.path.insert(0, 'C:\\Users\\juanf\\.conda\\envs\\gcnn_env\\Lib\site-packages\\torch_sparse\\_convert_cuda.pyd')
from torch_geometric.nn import GCNConv

def gsm_estimation(S,F,W):
        n = W.shape[0]
        I = sp.sparse.eye(n)
        #D = scipy.sparse.spdiags(np.sum(W,1).squeeze(),0, W.shape[0], W.shape[1])
        D_sq_inv = sp.sparse.spdiags(np.power(np.sum(W,1),-0.5).squeeze(),0, W.shape[0], W.shape[1])
        L = I - D_sq_inv * W * D_sq_inv

        Y = S@F
        zeros = coo_matrix((S.shape[0], S.T.shape[1]), dtype=np.float32)
        A = vstack( [hstack([L,S.T]), hstack([S,zeros.tocsr()])] )
        zeros_b = coo_matrix((S.T.shape[0],1), dtype=np.float32)
        F_hat = np.zeros((n,2)) 
        b_0 = vstack([zeros_b,Y[:,0].reshape((Y[:,0].shape[0],1))])
        F_hat[:,0] = spsolve(A, b_0)[0:n]
        b_1 = vstack([zeros_b,Y[:,1].reshape((Y[:,1].shape[0],1))])
        F_hat[:,1] = spsolve(A, b_1)[0:n]
        f_hat = 1-np.argmax(F_hat,axis=1)
        return f_hat

#from torch_geometric.nn import GCNConvCould not find module 'C:\Users\juanf\.conda\envs\gcnn_env\Lib\site-packages\torch_sparse\_convert_cuda.pyd' (or one of its dependencies). Try using the full path with constructor syntax.
def gcn_estimation(S,F,W,X,hidden_chs):
    """
    Parameters:
    S : sparse k by n sampling matrix, allowing the selection of training examples.
    F : original one hot encoded label matrix of shape (n, num_of_classes)
    W : graph adjancecy matrix of shape (n,n)¨
    X : feature vector matrix of shape (n,d)
    """

    n = W.shape[0] # number of nodes
    number_of_classes = F.shape[1]


    class GCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels,add_self_loops=True)
            self.conv2 = GCNConv(hidden_channels, out_channels,add_self_loops=True)

        def forward(self, x, edge_index, edge_weight):
            #x, edge_index = data.x, data.edge_index

            x = self.conv1(x, edge_index, edge_weight)
            #x = self.conv1(x, edge_index)
            x = functional.relu(x)
            #x = functional.dropout(x, p = 0.5, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            #x = self.conv2(x, edge_index)

            return functional.log_softmax(x, dim=1)

    #if X == None:
    # Set X as a n by n identity matrix
        #nnz_entry_indices = torch.from_numpy(np.vstack((np.arange(n).reshape((1,n)),
        #                                                np.arange(n).reshape((1,n)))))
        #nnz_entry_vals = torch.from_numpy(np.ones((1,n)))
        #X = torch.sparse_coo_tensor(indices=nnz_entry_indices,
        #                            values = nnz_entry_vals.squeeze(),
        #                            size=(n,n), dtype=torch.float)
    #    X = torch.tensor(np.eye(n),dtype =torch.float32)
    #else:
    X = torch.tensor(X,dtype =torch.float32)
    num_of_features = X.shape[1]

    model = GCN(num_of_features,hidden_chs,number_of_classes)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    output = find(W)
    edge_indices =torch.tensor(np.vstack((output[0],output[1])),dtype =torch.int64)
    edge_weights = torch.tensor(output[2],dtype = torch.float32)

    # training labels
    Y = S@F
    y_train = torch.from_numpy(1-np.argmax(Y,axis=1))

    #idx_train = find(S)[1] # nonzero column indices of S

    idx_train = (S@np.arange(n).reshape((n,1)).squeeze()).astype(int)

    

    #model(x=X,edge_index=edge_indices)

    model.train()
    for epoch in range(400):
        optimizer.zero_grad()
        out = model(X,edge_indices,edge_weights)
        loss = functional.nll_loss(out[idx_train,:], y_train)
        print(loss)
        loss.backward()
        optimizer.step()

    #pass
    model.eval()
    f_hat = model(X,edge_indices,edge_weights).argmax(dim=1)
    return f_hat.numpy()
