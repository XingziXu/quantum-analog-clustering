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


if __name__ == '__main__':
    T = 100 # define terminal time for testing
    h = 6.626 * 1e-34 # planck constant
    m = 9.11 * 1e-30 # mass of electron
    cnst = h / (2 * m)
    N = 100 # number of vertices
    u0 = torch.rand(1,N) # initial u values
    
    vals = torch.rand(int(N*(N+1)/2)) # values, assume all vertices are somewhat connected
    W = torch.zeros(N, N) # initialize weight matrix
    i, j = torch.triu_indices(N, N) # indices
    W[i, j] = vals # set upper triangle values
    W.T[i, j] = vals # set lower triangle values, undirected graph, so symmetrical
    diag_idx = torch.linspace(0,99,100).long()
    W[diag_idx,diag_idx] = 1 # set diagonal values to be one (connected to itself)
    
    
    w_rows = W.sum(0) # calculate the total weights of each node
    L = torch.zeros_like(W) # initialize graph Laplacian
    for idx, val in enumerate(w_rows): # normalize weight of each node
        L[idx,:] = -W[idx,:]/val # calculate graph Laplacian
    L[diag_idx,diag_idx] = 1 # the diagonal will be 1
    
    
    j = torch.complex(torch.Tensor([0]), torch.Tensor([1])) # define imaginary number j
    R = torch.eye(N) + j * (cnst * L) # define R matrix for evolution
    
    
    # method 1 for calculating u
    u1 = torch.complex(torch.zeros(T,N),torch.zeros(T,N)) # initialize matrix to hold schrodinger wave values
    u1[0,:] = u0 # define the initial values of our schrodinger wave u
    for t in range(1,T): # evolve in time
        u1[t,:] = (torch.matmul(R,u1[t-1,:].unsqueeze(1))).squeeze() # update for the next time
    
    
    # method 2 for calculating u
    u2 = torch.complex(torch.zeros(T,N),torch.zeros(T,N)) # initialize matrix to hold schrodinger wave values
    u2[0,:] = u0 # define the initial values of our schrodinger wave u
    eval, evec = torch.linalg.eig(L)
    beta = torch.ones_like(eval) + j * cnst * eval
    beta_mag = torch.norm(torch.view_as_real(beta),dim=1)
    beta_ang = torch.atan(torch.view_as_real(beta)[:,1]/torch.view_as_real(beta)[:,0])
    for t in range(1,T):
        for n in range(N):
            u2[t,:] = u2[t,:] + torch.dot(u2[0,:],evec[:,n]) * ((beta_mag[n] * torch.exp(j * beta_ang[n])) ** t) * evec[:,n]
        #mag = torch.pow(beta_mag, t).unsqueeze(0)
        #exp = torch.exp(j * t * beta_ang).unsqueeze(0)
        #current = torch.complex(torch.zeros(N,N),torch.zeros(N,N))
        #current[diag_idx,diag_idx] = (torch.matmul(u2[0,:].unsqueeze(0),evec) * mag * exp)
        #u2[t,:] = (current * evec).sum(0)
    
    
    # comparing u calculated with the two methods
    u1_mag = torch.norm(u1,dim=1)
    u2_mag = torch.norm(u2,dim=1)
    udiff_mag = torch.norm(u1-u2,dim=1)
    u1_ang = torch.norm(torch.atan(torch.view_as_real(u1)[:,:,1]/torch.view_as_real(u1)[:,:,0]), dim=1)
    u2_ang = torch.norm(torch.atan(torch.view_as_real(u2)[:,:,1]/torch.view_as_real(u2)[:,:,0]), dim=1)
    udiff_ang = u1_ang-u2_ang
    
    
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(u1_mag, label='Method 1', color='mediumturquoise')
    ax[0].plot(u2_mag, label='Method 2',color='slateblue')
    ax[0].plot(udiff_mag, label='Difference',color='tomato')
    ax[0].set_ylabel('Magnitude')
    ax[0].set_xlabel('Time')
    ax[1].plot(u1_ang, color='mediumturquoise')
    ax[1].plot(u2_ang, color='slateblue')
    ax[1].plot(udiff_ang, color='tomato')
    ax[1].set_ylabel('Angle')
    ax[1].set_xlabel('Time')
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center')
    ax[0].grid()
    ax[1].grid()
    plt.tight_layout()
    plt.savefig('/Users/xingzixu/clustering/canonical/schr/figure/evolution_big_m.pdf')