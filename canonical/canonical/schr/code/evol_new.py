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

    h = 6.626 * 1e-34 # planck constant
    m = 9.11 * 1e-34 # mass of electron
    dt = 1e-2
    num_t = 1000
    T = num_t * dt # define terminal time for testing
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
    
    
    # method 1 for calculating u
    u1 = torch.complex(torch.zeros(num_t,N),torch.zeros(num_t,N)) # initialize matrix to hold schrodinger wave values
    u1[0,:] = u0 # define the initial values of our schrodinger wave u
    for t in range(1,num_t): # evolve in time
        u1[t,:] = (torch.matmul(R,u1[t-1,:].unsqueeze(1))).squeeze() * dt # update for the next time
    
    
    
    # comparing u calculated with the two methods
    u1_mag = torch.norm(u1,dim=1)
    u1_ang = torch.norm(torch.atan(torch.view_as_real(u1)[:,:,1]/torch.view_as_real(u1)[:,:,0]), dim=1)
    
    
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(u1_mag, label='Method 1', color='mediumturquoise')
    ax[0].set_ylabel('Magnitude')
    ax[0].set_xlabel('Time')
    ax[1].plot(u1_ang, color='mediumturquoise')
    ax[1].set_ylabel('Angle')
    ax[1].set_xlabel('Time')
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    ax[0].grid()
    ax[1].grid()
    plt.tight_layout()
    plt.savefig('/Users/xingzixu/clustering/canonical/schr/figure/evolution_big_m.png')