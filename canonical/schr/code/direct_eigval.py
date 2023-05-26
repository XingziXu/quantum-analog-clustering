""" we solve for u(t) according to schrodinger equation in two ways,
one way is by u(t)=R^t u(0), where W is defined in the latex file,
another way is by u(t)=sum_{j=1}^N u(0)^T v_j(|beta_j|^t exp(itw_j))v_j,
where w_j is calculated using beta_j, as defined in the latex file,
v_j is the eigenvector of graph Laplacian L.
We want to test:
1. the u(t) calculated in these two ways match
2. u(t)'s magnitude is increaing temporally
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import random


if __name__ == '__main__':
    random.seed(3)
    T = 100 # define terminal time for testing
    h = 6.626 * 1e-34 # planck constant
    m = 9.11 * 1e-34 # mass of electron
    dt = 1e-1
    theta = (2 * m) / (h * dt)
    V = 1e-30
    N = 100 # number of vertices
    u0 = torch.rand(1,N) # initial u values
    
    vals = torch.rand(int(N*(N+1)/2)) # values, assume all vertices are somewhat connected
    W = torch.zeros(N, N) # initialize weight matrix
    i, j = torch.triu_indices(N, N) # indices
    W[i, j] = vals # set upper triangle values
    W.T[i, j] = vals # set lower triangle values, undirected graph, so symmetrical
    diag_idx = torch.linspace(0,N-1,N).long()
    W[diag_idx,diag_idx] = 1 # set diagonal values to be one (connected to itself)
    
    
    w_rows = W.sum(0) # calculate the total weights of each node
    L = torch.zeros_like(W) # initialize graph Laplacian
    for idx, val in enumerate(w_rows): # normalize weight of each node
        L[idx,:] = -W[idx,:]/val # calculate graph Laplacian
    L[diag_idx,diag_idx] = 1 # the diagonal will be 1
    
    
    j = torch.complex(torch.Tensor([0]), torch.Tensor([1])) # define imaginary number j
    R = torch.eye(N) * (1-j * (V * dt/h)) + j * (L / theta) # define R matrix for evolution
    
    # calculate eigenvalue of L
    eval_L, evec_L = np.linalg.eig(L.numpy())
    
    # directly calculate eigenvalue of R
    eval_R, evec_R = np.linalg.eig(R.numpy())
    
    # calculate eigenvalue of R using eigenvalue of L
    eval_R_1 = 1 + j.numpy() * (eval_L.real / theta - V * dt / h)
    
    # calculate upper bound of Re(eval_R)
    upper = (2 * h - V * theta * dt) / (h * theta)
    
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(eval_R.real,eval_R.imag, label='Direct Calculation')
    ax[0].scatter(eval_R_1.real,eval_R_1.imag, label='Through L')
    ax[0].axhline(y = upper, color = 'k', linestyle = '--', label='Upper Bound')
    ax[1].plot(abs((eval_R.real-eval_R_1.real))/abs(eval_R.real), label='Error of Real')
    ax[1].plot(abs((eval_R.imag-eval_R_1.imag))/abs(eval_R.imag), label='Error of Imaginary')
    ax[1].set_yscale('log')
    #handles, labels = ax.get_legend_handles_labels()
    ax[0].legend(loc='upper left')
    ax[1].legend()
    plt.tight_layout()
    ax[0].grid()
    ax[1].grid()
    plt.savefig('/Users/xingzixu/clustering/canonical/schr/figure/R_eval.pdf')