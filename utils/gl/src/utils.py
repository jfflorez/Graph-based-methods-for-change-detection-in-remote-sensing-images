""" This file contains a python implementation of several functions, originally developed by
    Vassilis Kalofolias in December, 2015. These are important for the implementation
    of Kalofolias' graph learning algorithm in Python.

    Date: May, 2021
    Author: Juan Felipe Florez-Ospina
    testing: main_test_squareform_sp    
    """



import numpy as np
import warnings
from numpy import linalg as LA
from scipy.sparse import coo_matrix, isspmatrix, find
from scipy.spatial.distance import squareform

def isvector(w):
    return w.ndim == 1 or 1 in np.shape(w)
def iscolumn(w):
    if w.ndim > 1:
        ncols = w.shape[1]
    else:
        ncols = 0
    return (ncols == 1)

def reshape_as_column(U):
    if U.ndim == 1:
        nrows = U.shape[0]
    else:
        nrows = U.shape[0]*U.shape[1]
    return U.reshape((nrows,1))

def nnz(w):
    output = find(reshape_as_column(w))
    return np.max(output[0].shape) # counts non zeros along flattened version of w
#def numel(w):
#    return np.size(w)

def squareform_sp(w):

    """SQUAREFORM_SP Sparse counterpart of matlab's squareform
    Usage: w = squareform_sp(W);

    Input parameters:
        w: sparse vector with n(n-1)/2 elements OR
        W: matrix with size [n, n] and zero diagonal

    Output parameters:
        W: matrix form of input vector w OR
        w: vector form of input matrix W

    This function is to be used instead of squareform.m when the matrix W
    or the vector w is sparse. For large scale computations, e.g. for
    learning the graph structure of a big graph it is necessary to take
    into account the sparsity.

    Example::

        B = sprand(8, 8, 0.1);
        B = B+B';
        B(1:9:end) = 0;
        b = squareform_sp(B);
        Bs = squareform_sp(b);


    See also: sum_squareform zero_diag

    Date: December 2015
    Author: Vassilis Kalofolias
    Testing: test_squareform"""

    # if input is not sparse it doesn't make sense to use this function!
    #if (not isspmatrix(w) ) & (nnz(w)/numel(w) > 1/10):
    if not isspmatrix(w):
        w = squareform(w.squeeze())
    #     warning('Use standard squareform for non sparse vector! Full version returned.');
        return w

    if isvector(w):
        # VECTOR -> MATRIX
        l = w.shape[0]
        n = round((1 + np.sqrt(1+8*l))/2)
        # check input
        if not(l == n*(n-1)/2):
            raise NameError('Bad vector size!')
        
        # ToDO: make more efficient! (Maybe impossible)
        if iscolumn(w):
            ind = find(w)
            ind_vec = ind[0]
            s = reshape_as_column(ind[2])
            #[ind_vec, ~, s] = find(w)
        else:
            ind = find(w)
            ind_vec = ind[1]
            s = reshape_as_column(ind[2])
            #[~, ind_vec, s] = find(w)
        
        
        num_nz = ind_vec.size
        
        # indices inside the matrix
        ind_i = np.zeros((num_nz, 1))
        ind_j = np.zeros((num_nz, 1))
        
        
        curr_row = 1
        offset = 0
        # length of current row of matrix, counting from after the diagonal
        len = n - 1
        for ii in range(num_nz): # ii = 1 : length(ind_vec):
            ind_vec_i = ind_vec[ii]+1 # we add 1 to python indices to comply with matlab's first index
            # if we change row, the vector index is bigger by at least the
            # length of the line + the offset.
            while ind_vec_i > (len + offset):
                offset = offset + len
                len = len - 1
                curr_row = curr_row + 1            
            ind_i[ii] = curr_row
            ind_j[ii] = ind_vec_i - offset + (n-len)

        ind_i = ind_i - 1
        ind_j = ind_j - 1
        
        # for the lower triangular part just add the transposed matrix
        # w = sparse(ind_i, ind_j, s, n, n) + sparse(ind_j, ind_i, s, n, n);
        # w = sparse([ind_i; ind_j], [ind_j; ind_i], [s(:); s(:)], n, n);
        w = coo_matrix((s.squeeze(), (ind_i.squeeze(), ind_j.squeeze())), shape=(n, n))
        w = w + w.T
        
    else:
        ## MATRIX -> VECTOR
        # first checks
        m = w.shape[0]
        n = w.shape[1]
        
        if (m != n) | (np.sum(w.diagonal(0)) != 0):
            raise NameError('Matrix has to be square with zero diagonal!')
        
        ind = find(w)
        ind_i = ind[0]
        ind_j = ind[1]
        s = ind[2]

        # keep only upper triangular part
        ind_upper = ind_i < ind_j
        ind_i = ind_i[ind_upper]
        ind_j = ind_j[ind_upper]
        s = s[ind_upper]
        
        # compute new (vector) index from (i,j) (matrix) indices
        # new_ind = (ind_j + (ind_i-1)*n - ind_i*(ind_i+1)/2).astype('int32')
        new_ind = (n*(n-1)/2 - (n-ind_i)*(n-ind_i-1)/2 + ind_j - ind_i - 1).astype('int32')
        w = coo_matrix((s, (new_ind, np.zeros(new_ind.shape).astype('int32'))), shape=(int(n*(n-1)/2), 1))
        #w = coo_matrix(s)
        # w = sparse(new_ind, 1, s, n*(n-1)/2, 1);
    
    return w

def sum_squareform(n, mask = []):
    #SUM_SQUAREFORM sparse matrix that sums the squareform of a vector
    #   Usage:  [S, St] = sum_squareform(n)
    #           [S, St] = sum_squareform(n, mask)
    #
    #   Input parameters:
    #         n:    size of matrix W
    #         mask: vector: np.array or sparse mtx. 
    #               if given, S only contain the columns indicated by the mask
    #
    #   Output parameters:
    #         S:    matrix such that S*w = sum(W,2), where w = squareform(W)
    #         St:   the adjoint of S
    #
    #   Creates sparse matrices S, St = S' such that
    #       S*w = sum(W,2),       where w = squareform(W)
    #
    #   The mask is used for large scale computations where only a few
    #   non-zeros in W are to be summed. It needs to be the same size as w,
    #   n(n-1)/2 elements. See the example below for more details of usage.
    #
    #   Properties of S:
    #   * size(S) = [n, (n(n-1)/2)]     % if no mask is given.
    #   * size(S, 2) = nnz(w)           % if mask is given
    #   * norm(S)^2 = 2(n-1)
    #   * sum(S) = 2*ones(1, n*(n-1)/2)
    #   * sum(St) = sum(squareform(mask))   -- for full mask = (n-1)*ones(n,1)
    #
    #   Example::
    #           % if mask is given, the resulting S are the ones we would get with the
    #           % following operations (but memory efficiently):
    #           [S, St] = sum_squareform(n);
    #           [ind_i, ~, w] = find(mask(:));
    #           % get rid of the columns of S corresponding to zeros in the mask
    #           S = S(:, ind_i);
    #           St = St(ind_i, :);
    #
    #   See also: squareform_sp
    #

    #
    # code author: Vassilis Kalofolias
    # date: June 2015
    # Testing: test_squareform

    n = int(n)
    
    if type(mask) == list:
        mask_given = False
    else:
        mask_given = True

    if mask_given:
        ## more efficient than the following:
        # M = squareform(mask);
        # [I, J] = find(triu(M));
        mask = reshape_as_column(mask)

        mask_length = np.max(mask.shape)        
        if not(mask_length == n*(n-1)/2):
            raise NameError('mask size has to be n(n-1)/2')
        
        
        # ToDO: make more efficient! (Maybe impossible)
        #if iscolumn(mask):
        output = find(mask)
        ind_vec = output[0]                    
        #else:
        #    output = find(mask)
        #    ind_vec = output[1]
        # Add 1 to make consistent with Matlab's indexing
        ind_vec = ind_vec + 1
        
        # final number of columns is the nnz(mask)
        ncols = np.max(ind_vec.shape)
        
        # indices inside the matrix
        I = np.zeros((ncols, 1))
        J = np.zeros((ncols, 1))
        
        curr_row = 1
        offset = 0
        # length of current row of matrix, counting from after the diagonal
        len = n - 1
        for ii in range(ncols):
            ind_vec_i = ind_vec[ii]
            # if we change row, the vector index is bigger by at least the
            # length of the line + the offset.
            while(ind_vec_i > (len + offset)):
                offset = offset + len
                len = len - 1
                curr_row = curr_row + 1
            
            I[ii] = curr_row
            J[ii] = ind_vec_i - offset + (n-len)
        # Remove 1 to make consitent with Python's indexing
        I = I - 1
        J = J - 1        
    else:
        ## more efficient than the following:
        # W = ones(n) - eye(n);
        # [I, J] = find(tril(W));
        
        # number of columns is the length of w given size of W
        ncols = int((n-1)*(n)/2)        
        I = np.zeros((ncols, 1))
        J = np.zeros((ncols, 1))
        
        # offset
        k = 1
        for i in range(2,n+1): # 2:n
            idx = np.arange(k, k + (n-i) + 1) 
            idx_length = idx.size           
            I[idx-1] = np.arange(i,n + 1).reshape((idx_length,1)) # I(k: k + (n-i)) = i : n
            J[idx-1] = (i-1)*np.ones((idx_length,1)) # J(k: k + (n-i)) = i-1; 
            k = k + (n-i+1)
        # Remove 1 to make consitent with Python's indexing
        I = I - 1
        J = J - 1  

        # k = 1
        # for i in range(2,n+1): # 2:n
        #    idx = np.arange(k, k + (n-i) + 1)  
        #    J[idx] = (i-1)*np.ones(idx.shape) # J(k: k + (n-i)) = i-1;            
        #    k = k + (n-i+1)
    
    II = np.concatenate((np.arange(1,ncols+1),np.arange(1,ncols+1)),axis=0).astype('int32')-1 # subtract 1 for consistency with Python indexing
    JJ = np.concatenate((I,J),axis=0).astype('int32').squeeze()


    St = coo_matrix((np.ones(II.shape), (II, JJ)), shape=(ncols, n))
    #St = sparse([1:ncols, 1:ncols], [I, J], 1, ncols, n)
    S = St.T

    if ncols > n:
        St.tocsc()
        #S.tocsr()
    else:
        St.tocsr()
        #S.tocsc()

    

    return S, St



def lin_map(X, lims_out, lims_in):
    """ LIN_MAP   Map linearly from a given range to another

    Input params:
        X: numpy array
        lims_out: list of two scalars
        lims_in: (optional) list of two scalars

    USAGE:
    Y = lin_map(X, lims_out, lims_in)
    
    X can be of any size. Elements of lims_in or lims_out don't have to be in
    an ascending order.

    if lims_in is not specified, the minimum and maximum values are used. 

    Example: 
        
    x = cos((1:50)/3) + .05*randn(1, 50);
    y = lin_map(x, [4, 2]);
    figure; plot(x, 'o'); hold all; plot(y, '*');

    code author: Vassilis Kalofolias
    date: Feb 2014"""

    #if np.isscalar(X):


    #if len(lims_in)==0 and (not np.isscalar(X)):
    #    lims_in = [np.min(reshape_as_column(X),axis=0), np.max(reshape_as_column(X),axis=0)] 
           

    a = lims_in[0]; b = lims_in[1]
    c = lims_out[0]; d = lims_out[1]

    if np.isscalar(X):
        Y = (X-a)*((d-c)/(b-a)) + c
    else:
        Y = (reshape_as_column(X)-a)*((d-c)/(b-a)) + c

    return Y

def main_test_squareform_sp():
    """ run tests on functions 'squareform_sp' and 'sum_squareform_sp'"""

    n = 4
    row  = np.array([0, 3, 1, 0])
    col  = np.array([0, 3, 1, 2])
    data = np.array([4, 5, 7, 9])
    A = np.array([[0,1,2,0],[0,0,3,0],[0,0,0,0],[0,0,0,0]])
    A = A + A.T
    #Z = coo_matrix((data, (row, col)), shape=(4, 4))
    Z = coo_matrix(A)  
    print('Original sparse matrix Z: \n', Z.toarray()) 
    z = squareform_sp(Z)
    print('Run squareform_sp(Z) -> z: : \n', z.toarray())
    Z = squareform_sp(z)
    print('Run squareform_sp(z) -> Z: : \n', Z.toarray())

    S, St = sum_squareform(n, mask = [])
    print('Compute S, and St')
    print('Run S*z: \n', (S@z).toarray())
    print('Run np.sum(Z,axis=1): \n', np.sum(Z,axis=1))

    mask_mtx = np.zeros((n,n))
    mask_mtx[0,1] = 1
    mask_mtx[1,0] = 1
    mask_mtx[0,2] = 1
    mask_mtx[2,0] = 1
    mask_mtx[1,2] = 1
    mask_mtx[2,1] = 1
    mask = squareform_sp(mask_mtx)
    S, St = sum_squareform(n, mask)
    print('Display S: \n', S.toarray())

