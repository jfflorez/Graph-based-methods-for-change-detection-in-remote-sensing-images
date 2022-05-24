""" This file contains a python implementation of several functions, originally developed by
    Vassilis Kalofolias in December, 2015. These are important for the implementation
    of Kalofolias' graph learning algorithm in Python.

    Date: May, 2021
    Author: Juan Felipe Florez-Ospina
    testing: main_test_squareform_sp    
    """
    
#from math import gamma
import time
import numpy as np
import utils.gl.src.utils as utils # explicit relative import

def prox_sum_log(x, gamma, param={'nargout': 1}):
    """    PROX_SUM_LOG Proximal operator of log-barrier  - sum(log(x))
        Usage:  sol = prox_sum_log(x, gamma)
                sol = prox_sum_log(x, gamma, param)
                [sol, info] = prox_sum_log(x, gamma, param)
        
        Input parameters:
                x     : Input signal (vector or matrix!).
                gamma : Regularization parameter.
                param : Structure of optional parameters.
        
        Output parameters:
                sol   : Solution.
                info : Structure summarizing informations at convergence
        
        PROX_SUM_LOG(x, gamma, param) solves:
        
            sol = argmin_{z} 0.5*||x - z||_2^2 - gamma * sum(log(z))
        
        param is a Matlab structure containing the following fields:
        
            param.verbose : 0 no log, (1) print -sum(log(z)), 2 additionally
            report negative inputs.
        
        MATRICES:
        Note that this prox works for matrices as well. The log of the sum
        gives the same result independently of which dimension we perform the
        summation over:
        
            sol = (x + sqrt(x.^2 + 4*gamma)) /2;
        
        info is a Matlab structure containing the following fields:
        
            info.algo : Algorithm used
            info.iter : Number of iteration
            info.time : Time of exectution of the function in sec.
            info.final_eval : Final evaluation of the function
            info.crit : Stopping critterion used 
        
        
        See also:  prox_l1 prox_l2 prox_tv prox_sum_log_norm2
        
        
        Url: https://epfl-lts2.github.io/unlocbox-html/doc/prox/prox_sum_log.php

        Copyright (C) 2012-2016 Nathanael Perraudin.
        This file is part of UNLOCBOX version 1.7.4
        
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

        Author: Vassilis Kalofolias
        Date: June 2015
        Testing: test_prox_sum_log"""

    # Start the time counter
    t1 = time.time()

    # if nargin < 3, param = struct; end

    if not 'verbose' in param: 
        param['verbose'] = 1
    if not 'nargout' in param:
        param['nargout'] = 1
    #if ~isfield(param, 'verbose'), param.verbose = 1; end


    # test the parameters
    # stop_error = test_gamma(gamma);
    if gamma < 0:
        raise NameError('Gamma can not be negative')
    elif gamma == 0:
        stop_error = True
    else:
        stop_error = False

    if stop_error:
        sol = x
        info = {'algo': 'prox_sum_log',
                'iter': 0,
                'final_eval': 0,
                'crit' : '--',
                'time': time.time() - t1}
        #info.algo = mfilename;
        #info.iter = 0;
        #info.final_eval = 0;
        #info.crit = '--';
        #info.time = toc(t1);
        return sol, info



    sol = (x + np.sqrt(x**2 + 4*gamma)) /2
    info = {'algo': 'prox_sum_log',
            'iter': 0,
            'final_eval': -gamma * np.sum(np.log(utils.reshape_as_column(x)),axis = 0),
            'crit' : '--',
            'time': time.time() - t1}
        
    # Log after the prox
    if param['verbose'] >= 1:
        print('prox_sum_log: - sum(log(x)) =', info['final_eval'] / gamma)
        if param['verbose'] > 1:
            n_neg = utils.nnz(utils.reshape_as_column((x <= 0).astype('int32')))
            if n_neg > 0:
                print('(',n_neg,' negative elements, log not defined, check stability)')
        print('\n')

    if param['nargout'] == 1: 
        return sol 
    else: 
        return sol, info



def main_test_prox_sum_log():
    m = 4
    x = np.random.rand(m,2)
    gamma = 1/2
    param = {'verbose': 2, 'nargout': 2}
    sol, info = prox_sum_log(x, gamma, param)
    print(sol)
    print(info)
    #z = pdist(X, 'euclidean') # distance matrix condensed form
    #Z = utils.squareform(z) # turns the condensed form into a 4 by 4 distance matrix
    #Z = coo_matrix(Z)

