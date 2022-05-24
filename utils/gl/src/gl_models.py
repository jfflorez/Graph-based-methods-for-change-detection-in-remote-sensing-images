""" This file contains a python implementation of the graph learning model, 'gsp_learn_graph_log_degrees', originally developed by
    Vassilis Kalofolias in December, 2015. These are important for the implementation
    of Kalofolias' graph learning algorithm in Python.

    Date: May, 2021
    Author: Juan Felipe Florez-Ospina
    testing: main_test_squareform_sp    
    """

import numpy as np
import scipy as sp
import time
from scipy.sparse import coo_matrix, isspmatrix, find
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

#from utils import *
import utils.gl.src.utils as utils
import utils.gl.src.prox  as prox

def gsp_learn_graph_log_degrees(Z, a, b, params={'nargout': 1}):

    """ GSP_LEARN_GRAPH_LOG_DEGREES learns an adjacency matrix W from a pairwise distance matrix Z by solving:
    
            minimize_W sum(sum(W .* Z)) - a * sum(log(sum(W))) + b * ||W||_F^2/2 + c * ||W-W_0||_F^2/2
                s.t. W being a non-negative, symmetric matrix with zero diagonal.

    sum(sum(W .* Z)) promotes the columns of a data matrix X to be smooth on the resulting graph, where Z is 
    the pairwise distance matrix of X (See Refs [1,3]). Forward-backward-forward (FBF) based primal dual
    optimization is used to solve the problem (See Ref [2]).
    
    Parameters:
            Z         : np.ndarray or sparse scipy array, 
            matrix (or vector <-> compressed matrix ) with (squared) pairwise distances of nodes
            a         : float, log prior constant  (bigger a -> bigger weights in W)
            b         : float, ||W||_F^2 prior constant  (bigger b -> more dense W)
            params    : dict, optional parameters
    Returns:
            W         : scipy.sparse.coo_matrix, weighted adjacency matrix
            if params['nargout'] > 1
            stat      : dict, optional output statistics (adds small overhead)

    Usage:  W = gsp_learn_graph_log_degrees(Z, a, b)
            params = {'nargout': 2}
            W, stat = gsp_learn_graph_log_degrees(Z, a, b, params)

    Example:
            from pygsp import graphs
            G = gsp_sensor(256);
            f1 = lambda x, y: np.sin((2-x-y)**2);
            f2 = lambda x, y: np.cos((x+y)**2);
            f3 = lambda x, y: (x-.5)**2 + (y-.5)**3 + x - y;
            f4 = lambda x, y: np.sin(3*((x-.5)**2+(y-.5)**2));
            X = [f1(G.coords(:,1), G.coords(:,2)), f2(G.coords(:,1), G.coords(:,2)), f3(G.coords(:,1), G.coords(:,2)), f4(G.coords(:,1), G.coords(:,2))];
            figure; subplot(2,2,1); gsp_plot_signal(G, X(:,1)); title('1st smooth signal');
            subplot(2,2,2); gsp_plot_signal(G, X(:,2)); title('2nd smooth signal');
            subplot(2,2,3); gsp_plot_signal(G, X(:,3)); title('3rd smooth signal');
            subplot(2,2,4); gsp_plot_signal(G, X(:,4)); title('4th smooth signal');
            Z = gsp_distanz(X').^2;
            % we can multiply the pairwise distances with a number to control sparsity
            [W] = gsp_learn_graph_log_degrees(Z*25, 1, 1);
            % clean up zeros
            W(W<1e-5) = 0;
            G2 = gsp_update_weights(G, W);
            figure; gsp_plot_graph(G2); title('Graph with edges learned from above 4 signals');

    Additional parameters
    ---------------------

        params.W_init   : Initialization point. default: zeros(size(Z))
        verbosity       : Default = 1. Above 1 adds a small overhead
        maxit           : Maximum number of iterations. Default: 1000
        tol             : Tolerance for stopping criterion. Defaul: 1e-5
        step_size       : Step size from the interval (0,1). Default: 0.5
        max_w           : Maximum weight allowed for each edge (or inf)
        w_0             : Vector for adding prior c/2*||w - w_0||^2
        c               : multiplier for prior c/2*||w - w_0||^2 if w_0 given
        fix_zeros       : Fix a set of edges to zero (true/false)
        edge_mask       : Mask indicating the non zero edges if "fix_zeros"

    If fix_zeros is set, an edge_mask is needed. Only the edges
    corresponding to the non-zero values in edge_mask will be learnt. This
    has two applications: (1) for large scale applications it is cheaper to
    learn a subset of edges. (2) for some applications we don't want some
    connections to be allowed, for example for locality on images.

    The cost of each iteration is linear to the number of edges to be
    learned, or the square of the number of nodes (numel(Z)) if fix_zeros
    is not set.

    The function is using the UNLocBoX functions sum_squareform and
    squareform_sp.
    The stopping criterion is whether both relative primal and dual
    distance between two iterations are below a given tolerance.

    To set the step size use the following rule of thumb: Set it so that
    relative change of primal and dual converge with similar rates (use
    verbosity > 1).

    See also: gsp_learn_graph_l2_degrees gsp_distanz gsp_update_weights
        squareform_sp sum_squareform gsp_compute_graph_learning_theta

    References:
        [1] V. Kalofolias. How to learn a graph from smooth signals. Technical
        report, AISTATS 2016: proceedings at Journal of Machine Learning
        Research (JMLR)., 2016.
        
        [2] N. Komodakis and J.-C. Pesquet. Playing with duality: An overview of
        recent primal? dual approaches for solving large-scale optimization
        problems. Signal Processing Magazine, IEEE, 32(6):31--54, 2015.
        
        [3] V. Kalofolias and N. Perraudin. Large Scale Graph Learning from Smooth
        Signals. arXiv preprint arXiv:1710.05654, 2017.
        


    Url: https://epfl-lts2.github.io/gspbox-html/doc/learn_graph/gsp_learn_graph_log_degrees.html

    Copyright (C) 2013-2016 Nathanael Perraudin, Johan Paratte, David I Shuman.
    This file is part of GSPbox version 0.7.5

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    If you use this toolbox please kindly cite
        N. Perraudin, J. Paratte, D. Shuman, V. Kalofolias, P. Vandergheynst,
        and D. K. Hammond. GSPBOX: A toolbox for signal processing on graphs.
        ArXiv e-prints, Aug. 2014.
    http://arxiv.org/abs/1408.5781


    Author: Vassilis Kalofolias
    Testing: gsp_test_learn_graph
    Date: June 2015"""

    # Default parameters
    #if nargin < 4:
    #    params = {}

    if not (isinstance(Z,np.ndarray) or isspmatrix(Z)):
        raise ValueError('Z must be either a np.ndarray or scipy sparse array of dimension 1 or 2.')

    if not sum(utils.reshape_as_column(Z)) > 0:
        raise ValueError('Z cannot be all zeros.')

    if not 'verbosity' in params: params['verbosity'] = 1 
    if not 'maxit' in params:     params['maxit'] = 1000
    if not 'tol' in params:       params['tol'] = 1e-5
    if not 'step_size' in params: params['step_size'] = .5 # from (0, 1)
    if not 'fix_zeros' in params: params['fix_zeros'] = False 
    if not 'max_w' in params:     params['max_w'] = np.Inf 
    if not 'nargout' in params: params['nargout'] = 1
    
    #print(params.values())

    ## Fix parameter size and initialize
    if utils.isvector(Z):
        z = Z;  # lazy copying of matlab doesn't allocate new memory for z
    else:
        z = utils.squareform_sp(Z)
    # clear Z   # for large scale computation

    # Check if Z is a compressed form of a n by n distance matrix
    card_E_0 = np.max(np.shape(z)) # initial number of edges
    n = np.round((1 + np.sqrt(1+8*card_E_0))/ 2); # number of nodes
    # n(n-1)/2 = l => n = (1 + sqrt(1+8*l))/ 2
    if not (card_E_0 - n*(n-1)/2 == 0):
        raise ValueError('The length of Z must be the same as the number of upper diagonal elements of an n x n matrix')

    z = utils.reshape_as_column(z)

    if 'w_0' in params:
        if not 'c' in params: 
            raise NameError('When params.w_0 is specified, params.c should also be specified')
        else:
            c = params['c']
        if utils.isvector(params['w_0']):
            w_0 = params['w_0']
        else:
            w_0 = utils.squareform_sp(params['w_0'])
        w_0 = utils.reshape_as_column(w_0) 
    else:
        w_0 = np.zeros(z.shape)

    # if sparsity pattern is fixed we optimize with respect to a smaller number
    # of variables, all included in w
    if params['fix_zeros']:
        if not utils.isvector(params['edge_mask']):
            params['edge_mask'] = utils.squareform_sp(params['edge_mask'])
        # use only the non-zero elements to optimize
        ind = find(utils.reshape_as_column(params['edge_mask'])) # ind is a tuple of arrays (I,J,Val)
        ind = ind[0]
        z = z[ind]
        if not np.isscalar(w_0):
            w_0 = w_0[ind]
    else:
        # Make z and w_0 full matrices
        if isspmatrix(z): z = z.toarray() # full(z)
        if isspmatrix(w_0): w_0 = w_0.toarray() # full(w_0)

    w = np.zeros((z.shape))

    ## Needed operators
    # S*w = sum(W)
    if params['fix_zeros']:
        [S, St] = utils.sum_squareform(n,utils.reshape_as_column(params['edge_mask']))
    else:
        [S, St] = utils.sum_squareform(n)

    # S: edges -> nodes
    K_op = lambda w: S@w # K_op = @(w) S*w

    # S': nodes -> edges
    Kt_op = lambda z: St@z # Kt_op = @(z) St*z

    if params['fix_zeros']:
        #norm_K = norm(S,2) ## commented out and enable the below approximation. sp.sparse.linalg.norm() does not work, raise NotImplementedError
        # approximation: 
        norm_K = np.sqrt(2*(n-1)) * np.sqrt(utils.nnz(params['edge_mask']) / (n*(n+1)/2)) /np.sqrt(2)
    else:
        # the next is an upper bound if params.fix_zeros
        norm_K = np.sqrt(2*(n-1))

    ## ToDO: Rescaling??
    # we want    h.beta == norm_K   (see definition of mu)
    # we can multiply all terms by s = norm_K/2*b so new h.beta==2*b*s==norm_K


    ## Learn the graph
    # min_{W>=0}     tr(X'*L*X) - gc * sum(log(sum(W))) + gp * norm(W-W0,'fro')^2, where L = diag(sum(W))-W
    # min_W       I{W>=0} + W(:)'*Dx(:)  - gc * sum(log(sum(W))) + gp * norm(W-W0,'fro')^2
    # min_W                f(W)          +       g(L_op(W))      +   h(W)

    # put proximal of trace plus positivity together

    f = {'eval': lambda w: 2*np.transpose(w)@z, 
         'prox': lambda w, c: np.minimum(params['max_w'], np.maximum(0,w - 2*c*z))
         }
    #f.eval = @(w) 2*w'*z;    # half should be counted
    #%f.eval = @(W) 0;
    #f.prox = @(w, c) min(params.max_w, max(0, w - 2*c*z));  % all change the same

    param_prox_log = {'verbose': params['verbosity'] - 3}
    g = {'eval': lambda z: -a*np.sum(np.log(z),axis=0),
         'prox': lambda z, c: prox.prox_sum_log(z, c*a, param_prox_log)}
    #g.eval = @(z) -a * sum(log(z));
    #g.prox = @(z, c) prox_sum_log(z, c*a, param_prox_log);
    # proximal of conjugate of g: z-c*g.prox(z/c, 1/c)
    g_star_prox = lambda z, c: z - c*a*prox.prox_sum_log(z/(c*a), 1/(c*a), param_prox_log)

    if np.sum(w_0) == 0:
        # "if" not needed, for c = 0 both are the same but gains speed
        h = {'eval': lambda w: b*np.linalg.norm(w,2)**2,
             'grad': lambda w: 2*b*w,
             'beta': 2*b}
        #h.eval = @(w) b * norm(w)^2;
        #h.grad = @(w) 2 * b * w;
        #h.beta = 2 * b;
    else:
        h = {'eval': lambda w: b*np.linalg.norm(w,2)**2 + c * np.linalg.norm(w - w_0,'fro')**2,
             'grad': lambda w: 2*((b+c) * w - c * w_0),
             'beta': 2*(b+c)}
        #h.eval = @(w) b * norm(w)^2 + c * norm(w - w_0,'fro')^2
        #h.grad = @(w) 2 * ((b+c) * w - c * w_0);
        #h.beta = 2 * (b+c);

    ## My custom FBF based primal dual (see [1] = [Komodakis, Pesquet])
    # parameters mu, epsilon for convergence (see [1])
    mu = h['beta'] + norm_K;     #ToDO: is it squared or not??
    epsilon = utils.lin_map(0.0, [0, 1/(1+mu)], [0,1]);   # in (0, 1/(1+mu) )


    # INITIALIZATION
    # primal variable ALREADY INITIALIZED
    # w = params.w_init;
    # dual variable
    v_n = K_op(w)
    #if nargout > 1 || params['verbosity'] > 1:
    stat = {} # define empty dictionary
    if (params['nargout'] > 1) | (params['verbosity'] > 1):
        stat['f_eval'] = np.empty((params['maxit'], 1)); stat['f_eval'][:] = np.nan
        stat['g_eval'] = np.empty((params['maxit'], 1)); stat['g_eval'][:] = np.nan
        stat['h_eval'] = np.empty((params['maxit'], 1)); stat['h_eval'][:] = np.nan
        stat['fgh_eval'] = np.empty((params['maxit'], 1)); stat['fgh_eval'][:] = np.nan
        stat['pos_violation'] = np.empty((params['maxit'], 1)); stat['pos_violation'][:] = np.nan
    
    if params['verbosity'] > 1:
        print('Relative change of primal, dual variables, and objective fun\n')

    t1 = time.time()
    gn = utils.lin_map(params['step_size'], [epsilon, (1-epsilon)/mu], [0,1]) # in [epsilon, (1-epsilon)/mu]
    for i  in range(params['maxit']): # i = 1:params.maxit
        Y_n = w - gn * (h['grad'](w) + Kt_op(v_n))
        y_n = v_n + gn * (K_op(w))
        P_n = f['prox'](Y_n, gn)
        p_n = g_star_prox(y_n, gn); # = y_n - gn*g_prox(y_n/gn, 1/gn)
        Q_n = P_n - gn * (h['grad'](P_n) + Kt_op(p_n))
        q_n = p_n + gn * (K_op(P_n))
        
        if (params['nargout'] > 1) | (params['verbosity'] > 2):
            stat['f_eval'][i] = f['eval'](w) # f['eval'] contains a lambda function, and the same logic follows in the next few lines
            stat['g_eval'][i] = g['eval'](K_op(w))
            stat['h_eval'][i] = h['eval'](w)
            stat['fgh_eval'][i] = stat['f_eval'][i] + stat['g_eval'][i] + stat['h_eval'][i]
            stat['pos_violation'][i] = -np.sum(np.minimum(0,w),axis=0)

        rel_norm_primal = np.linalg.norm(-Y_n + Q_n,'fro')/np.linalg.norm(w,'fro')
        rel_norm_dual = np.linalg.norm(- y_n + q_n,2)/np.linalg.norm(v_n,2)
        
        if params['verbosity'] > 3:
            print('iter', i, ':', rel_norm_primal, rel_norm_dual, stat['fgh_eval'][i])
        elif params['verbosity'] > 2:
            print('iter', i, ':', rel_norm_primal, rel_norm_dual, stat['fgh_eval'][i])
        elif params['verbosity'] > 1:
            print('iter', i, ':', rel_norm_primal, rel_norm_dual)
        
        w = w - Y_n + Q_n
        v_n = v_n - y_n + q_n
        
        if (rel_norm_primal < params['tol']) & (rel_norm_dual < params['tol']):
            break
    stat['time'] = time.time() - t1

    if params['verbosity'] > 0:
        print('# iters: ',i,' Rel primal: ', rel_norm_primal, ' Rel dual: ', rel_norm_dual,' OBJ: ', f['eval'](w) + g['eval'](K_op(w)) + h['eval'](w),'\n')        
        print('Time needed is ',stat['time'],' seconds\n')

    # use the following for testing:
    # g.L = K_op;
    # g.Lt = Kt_op;
    # g.norm_L = norm_K;
    # [w, info] = fbf_primal_dual(w, f, g, h, params);
    # #[w, info] = fb_based_primal_dual(w, f, g, h, params);

    
    if params['verbosity'] > 3:
        plt.figure()
        ax0 = plt.gca()
        for i in ['f_eval','h_eval','g_eval','fgh_eval']:
            ax0.plot(stat[i],label=i)
        ax0.set_xlabel('iterations', fontsize=15)
        ax0.set_ylabel('function value', fontsize=15)
        lgd = ax0.legend(loc='best',ncol = 1,fontsize=12)   
        #ax0.plot(real([stat.f_eval, stat.g_eval, stat.h_eval])); hold all; plot(real(stat.fgh_eval), '.'); legend('f', 'g', 'h', 'f+g+h');
    #    plt.figure()
    #    plot(stat.pos_violation); title('sum of negative (invalid) values per iteration')
    #    plt.figure()
    #    semilogy(max(0,-diff(real(stat.fgh_eval'))),'b.-'); hold on; semilogy(max(0,diff(real(stat.fgh_eval'))),'ro-'); title('|f(i)-f(i-1)|'); legend('going down','going up');

    if params['fix_zeros']:
        #w = sparse(ind, ones(size(ind)), w, l, 1)
        w = coo_matrix((w.squeeze(), (ind, np.zeros(ind.shape))), shape=(card_E_0, 1))

    if utils.isvector(Z):
        W = w
    else:
        W = utils.squareform_sp(w)

    if params['nargout'] > 1:
        return W, stat
    else:
        return W

def main_test_gsp_learn_graph_log_degrees():

    #row  = np.array([0, 3, 1, 0])
    #col  = np.array([0, 3, 1, 2])
    #data = np.array([4, 5, 7, 9])
    #Z = coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    #z = Z.reshape((16,1))

    # Define 2-dim features on a graph with 4 vertices
    m = 10
    X = np.random.rand(m,2)
    z = pdist(X, 'euclidean') # distance matrix condensed form
    Z = utils.squareform(z) # turns the condensed form into a 4 by 4 distance matrix
    #Z = coo_matrix(Z)

    params = {}
    params['w_0'] = np.zeros((m,m))
    params['c'] = 1
    params['fix_zeros'] = 5
    params['edge_mask'] = np.zeros((m,m)) #np.reshape(np.array([1,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0]),(16,1))
    params['edge_mask'][Z>0.5] = 1
    params['verbosity'] = 3
    params['nargout'] = 1 

    nargout = 1
    a = 1
    b = 1

    W = gsp_learn_graph_log_degrees(Z,a,b,params)
    plt.figure()
    ax = plt.gca()
    ax.imshow(W.toarray())


    a = 0

def estimate_theta(Z,k):

    n = Z.shape[0]

    theta_ub = 0
    theta_lb = 0
    for i in range(n):
        z_row = Z[i,:]
        idx_sort = np.argsort(z_row)
        #b_k = np.sum(z[idx_sort[0:card_E]])
        b_k = np.cumsum(z_row[idx_sort])[k-1]
        z_k = z_row[idx_sort[k-1]]
        z_k_plus_1 = z_row[idx_sort[k]]
        theta_ub += 1/np.sqrt(k*(z_k**2)-b_k*z_k)
        theta_lb += 1/np.sqrt(k*(z_k_plus_1**2)-b_k*z_k_plus_1)

    return np.sqrt((theta_lb*theta_ub))/n
