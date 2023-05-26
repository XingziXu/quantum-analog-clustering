""" we solve for \hat(u)(t) using two ways, the ground truth way
using the \hat(R) for propagation, and the approximation using
u(t) based on R, and then multiply the matrix exponential of \Delta L,
which is approximated using Taylor series. We want to investigate the
error in doing these two approximations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import random


if __name__ == '__main__':
    random.seed(3)
    j_img = torch.complex(torch.Tensor([0]), torch.Tensor([1])) # define imaginary number j
    T = 100 # define terminal time for testing
    h = 6.626 * 1e-34 # planck constant
    m = 10. * h#9.11 * 1e-34 # mass of electron
    dt = m / h
    V = - j_img * h / dt
    N = 100 # number of vertices
    u0 = torch.rand(1,N) # initial u values
    num_t = 80
    
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
    R = h / (2 * m) * L - j_img / dt * torch.eye(N) # define R matrix for evolution
    
    ut = torch.complex(torch.zeros(num_t, N), torch.zeros(num_t, N))
    ut[0,:] = u0
    for idx_t in range(1, num_t):
        ut[idx_t,:] = ut[idx_t-1,:] - j_img * dt * torch.matmul(R, ut[idx_t-1,:].unsqueeze(1)).squeeze()
    
    dW = torch.zeros(N, N)
    dW[3,5] = W[3,5] * 0.2
    dW[5,3] = W[5,3] * 0.2
    dW[47,62] = W[47,62] * 0.3
    dW[62,47] = W[62,47] * 0.3
    dW[33,44] = W[33,44] * 0.6
    dW[44,33] = W[44,33] * 0.6
    dW[28,49] = W[28,49] * 0.6
    dW[49,28] = W[49,28] * 0.6
    W_hat = W - dW
    w_rows_hat = W_hat.sum(0) # calculate the total weights of each node
    L_hat = torch.zeros_like(W_hat) # initialize graph Laplacian
    for idx, val in enumerate(w_rows_hat): # normalize weight of each node
        L_hat[idx,:] = -W_hat[idx,:]/val # calculate graph Laplacian
    L_hat[diag_idx,diag_idx] = 1 # the diagonal will be 1
    R_hat = h / (2 * m) * L_hat - j_img / dt * torch.eye(N) # define R matrix for evolution
    
    ut_hat = torch.complex(torch.zeros(num_t, N), torch.zeros(num_t, N))
    ut_hat[0,:] = u0
    for idx_t in range(1, num_t):
        ut_hat[idx_t,:] = ut_hat[idx_t-1,:] - j_img * dt * torch.matmul(R_hat, ut_hat[idx_t-1,:].unsqueeze(1)).squeeze()
    
    dL = L_hat - L
    #dL_pow = torch.zeros(N, N, 20)
    
    #for i in range(0,19):
    #    dL_pow[:,:,i] = torch.pow(dL,i)
    
    ut_tilde = torch.zeros_like(ut_hat)
    ut_tilde[0,:] = u0
    for t_idx in range(1,num_t-1):
        dL_t = torch.zeros_like(dL)
        for k in (0,19):
            dL_t = dL_t + torch.matrix_power(-j_img * t_idx * dt * dL,k)/(np.math.factorial(k))
        #dL_t = torch.matrix_exp(-j_img * t_idx * dt * dL)
        ut_tilde[t_idx,:] = torch.matmul(dL_t, ut[t_idx,:].unsqueeze(1)).squeeze()
    
    diff = ut_tilde - ut_hat
    ut_hat_mag = torch.sqrt(ut_hat.real ** 2 + ut_hat.imag ** 2)
    diff_mag = torch.sqrt(diff.real ** 2 + diff.imag ** 2)
    diff_perc = diff_mag / ut_hat_mag
    diff_norm = torch.norm(diff_mag, dim=1)/torch.norm(ut_hat_mag)
    diff_ang = torch.atan(diff.imag/diff.real)
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(diff_norm, label='Normalized error')
    #ax[0].plot(torch.norm(ut_tilde[0:20],dim=1), label='Norm of ground truth')
    #ax[0].plot(torch.norm(ut_hat[0:20],dim=1), label='Norm of approximation')
    #ax[0].plot(torch.norm(ut,dim=1), label='Norm of original')
    ax[1].plot(ut_tilde[3,:])
    ax[1].plot(ut_hat[3,:])
    ax[0].legend(loc='upper left')
    #ax[0].legend()
    plt.tight_layout()
    ax[0].grid()
    ax[1].grid()
    plt.savefig('/Users/xingzixu/clustering/dynamic/schr/figure/err_analysis.png')