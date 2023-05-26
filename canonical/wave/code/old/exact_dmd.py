import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos, cosh, tanh, real, imag
from numpy.linalg import inv, eig, pinv, solve
from scipy.linalg import svd, svdvals
from scipy.integrate import odeint, ode, complex_ode
from warnings import warn
import torch
import scipy as sp
import random
#from numpy.linalg import lstsq


def ExactDMD(XX):
    '''
    Exact and standard DMD of the data matrices X and Y.

    :param mode: 'exact' for exact DMD or 'standard' for standard DMD
    :return:     eigenvalues d and modes Phi
    
    '''
    
    X = XX[:-1, :]
    Y = XX[1:, :]
    U, s, Vt = sp.linalg.svd(X, full_matrices=False)
    mode = 'exact'
    which="LM"
    svThresh=1.e-10
    # remove zeros from s
    s = s[s > svThresh]
    r = len(s)
    U = U[:, :r]
    Vt = Vt[:r, :]
    S_inv = sp.diag(1/s)
    A = U.T @ Y @ Vt.T @ S_inv
    # d, W = sortEig(A, A.shape[0])
    # d, W = sortEig(A, r)
    n = A.shape[0]
    if r < n:
        d, W = sp.sparse.linalg.eigs(A, r, which=which)
    else:
        d, V = sp.linalg.eig(A)
        ind = d.argsort()[::-1]
        d = d[ind]
        W = V[:, ind]

    if mode == 'exact':
        Phi = Y @ Vt.T @ S_inv @ W @ sp.diag(1/d)
    elif mode == 'standard':
        Phi = U @ W
    elif mode == 'eigenvector':
        Phi = W
    else:
        raise ValueError('Only exact and standard DMD available.')

    return d, Phi


def dmd(X, Y, truncate=None):
    U2,Sig2,Vh2 = svd(X, False) # SVD of input matrix
    r = len(Sig2) if truncate is None else truncate # rank truncation
    U = U2[:,:r]
    Sig = diag(Sig2)[:r,:r]
    V = Vh2.conj().T[:,:r]
    Atil = dot(dot(dot(U.conj().T, Y), V), inv(Sig)) # build A tilde
    mu,W = eig(Atil)
    Phi = dot(dot(dot(Y, V), inv(Sig)), W) # build DMD modes
    Phitil = dot(Phi, diag(1/mu))
    atil = solve(Phitil, X[:,0])
    ai = Phitil[:,1] * atil
    return ai


if __name__ == '__main__':
    random.seed(3)
    N = 20 # number of vertices
    T = 2 * N # define terminal time for testing
    u0 = torch.rand(1,N) # initial u values
    c = np.sqrt(2) # wave speed
    
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
    
    u = torch.complex(torch.zeros(T+1,N),torch.zeros(T+1,N)) # initialize matrix to hold wave values
    u[0,:] = u0 # define u(-1)
    u[1,:] = u0 # define u(0)
    for i in range(0,N):
        for t in range(2,T+1): # evolve in time for u(1) to u(T-1)
            L_sum = L[:,i] * u[t-1,:]
            L_sum[i] = 0
            u[t,:] = 2 * u[t-1,:] - u[t-2,:] - (c ** 2) * torch.sum(L_sum)  # update for the next time
    
    
    X = torch.zeros(N, N, N) # initialize X values, last idx is for the node indices
    Y = torch.zeros_like(X) # initialize Y values, last idx is for the node indices
    a = torch.zeros(N,N) # intialize a values, last idx is for the node indices
    for i in range(0,N): # iterate through each node
        for n in range(0,N): # form the X, Y matrices and calculate the a values, according to algorithm 2.1
            X[n,:,i] = u[n:n+N,i] # form X matrix
            Y[n,:,i] = u[n+1:n+N+1,i] # form Y matrix
        a[:,i] = torch.Tensor(dmd(X[:,:,i],Y[:,:,i])) # calculate vector a

    #d, Phi = ExactDMD(u.numpy())
    #atil = lstsq(Phi.T, u[:N,:].numpy())[0]
    #a = Phi[:,1] * atil
    
    v = a # a is approximately the eigenvector of L
    A = torch.zeros(N,N)
    A[v>0] = 1
    cluster = (A * (2 ** torch.linspace(0,19,20)).unsqueeze(-1).repeat(1,N)).sum(dim=0)
    
    mu,W = eig(L)
    plt.plot(np.sign(W[:,2]), label='Laplacian $2^{nd}$ eigenvector')
    plt.plot(np.sign(v[2,:]), label='DMD')
    plt.legend()
    plt.savefig('/Users/xingzixu/clustering/canonical/wave/figure/dmd.png')