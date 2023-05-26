import numpy as np
import pandas as pd
import scipy.io
import scipy as sp
import matplotlib.pyplot as plt
from wave_dmd import (graph_laplacian_eigs, generate_time_sequece, append_time_sequence,
                              graph_clustering_static, graph_clustering_dynamic)

if __name__ == "__main__":
    np.random.seed(1234)
    plt.rcParams.update({"font.size": 28, "lines.linewidth": 3})

    evs = 3
    v_inds = range(1, evs)
    T = 49
    Nrows = 25
    base = "karate"
    case = f"DMD_karate_T{T}_rows{Nrows}"
    print(case)

    # load graph data
    data = scipy.io.loadmat('/Users/xingzixu/clustering/canonical/schr/data/karate.mat', squeeze_me=True)
    for s in data.keys():
        if s[:2] == '__' and s[-2:] == '__': continue
        exec('%s = data["%s"]' % (s, s))

    # Adjacent matrix and Laplacian for static graph
    N = np.max(A[:, :2])
    AdjMt0 = np.zeros((N, N))
    # AdjMt = np.zeros((N, N))
    L0 = np.zeros((N, N))
    for ii in range(A.shape[0]):
        AdjMt0[A[ii, 0] - 1][A[ii, 1] - 1] = 1
    AdjM = (AdjMt0 + AdjMt0.T) / 2.

    L, d_M, V_M, d_L, V_L = graph_laplacian_eigs(AdjM=AdjM, N=N, evs=evs)
    freq = np.angle(d_M[:3])
    print(f"Actual Freq 2: {freq}")

    u = np.zeros((N, T + 1))
    # initialize u(0) and set u(1)=u(0)
    for ii in range(N):
        u[ii, 0] = np.random.rand(1, 1)
        u[ii, 1] = u[ii, 0] 
    # df_u = pd.DataFrame(u[:, 0], columns=["u"])
    # df_u.to_csv("u_init.csv", index=False)
    c = np.sqrt(2.) - 1.e-6

    # generate time sequence for each node
    for tt in range(1, T):
        # for kk in range(N):
        #     tmp = 0
        #     for ll in range(N):
        #         if L[kk, ll]:
        #             tmp += L[kk, ll] * u[ll, tt]
        #     u[kk, tt+1] = 2.*u[kk, tt] - u[kk, tt-1] + c**2*tmp
        u[:, tt + 1] = 2. * u[:, tt] - u[:, tt - 1] + c ** 2 * L @ u[:, tt]
    
    w = np.zeros((2*N, T+1))
    w[:,0] = np.concatenate((u[:,0],np.zeros_like(u[:,0])),axis=0)
    prop_L = np.concatenate((np.concatenate((np.eye(N)+c**2 * L, np.eye(N)),axis=1),np.concatenate((+c**2 * L, np.eye(N)),axis=1)),axis=0)
    for tt in range(0, T):
        # for kk in range(N):
        #     tmp = 0
        #     for ll in range(N):
        #         if L[kk, ll]:
        #             tmp += L[kk, ll] * u[ll, tt]
        #     u[kk, tt+1] = 2.*u[kk, tt] - u[kk, tt-1] + c**2*tmp
        w[:, tt + 1] = prop_L @ w[:, tt]
    w[:,-1] == u[:,-1]