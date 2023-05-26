#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.io
import scipy as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from pydmd import DMD
import numpy.linalg as la

# import d3s.algorithms as algorithms

import pandas as pd


def dmd(X, Y, mode='exact', svThresh=1.e-10, which="SI"):
    '''
    Exact and standard DMD of the data matrices X and Y.

    :param mode: 'exact' for exact DMD or 'standard' for standard DMD
    :return:     eigenvalues d and modes Phi
    '''
    U, s, Vt = la.svd(X, full_matrices=False)
    # remove zeros from s
    s = s[s > svThresh]
    #s = s[np.argpartition(s,-5)[-5:]]
    r = len(s)
    U = U[:, :r]
    Vt = Vt[:r, :]
    S_inv = sp.diag(1/s)
    Atil = U.T.conj() @ Y @ Vt.T.conj() @ S_inv
    # d, W = sortEig(A, A.shape[0])
    # d, W = sortEig(A, r)
    n = Atil.shape[0]
    if r < n:
        d, W = sp.sparse.linalg.eigs(Atil, r, which=which)
    else:
        d, V = sp.linalg.eig(Atil)
        ind = d.argsort()[::-1]
        d = d[ind]
        W = V[:, ind]

    if mode == 'exact':
        Phi = Y @ Vt.T.conj() @ S_inv @ W @ sp.diag(1/d)
    elif mode == 'standard':
        Phi = U @ W
    elif mode == 'eigenvector':
        Phi = W
    else:
        raise ValueError('Only exact and standard DMD available.')

    return d, Phi


def graph_laplacian_eigs(AdjM, N, evs):
    D = np.diag(np.sum(AdjM,axis=1))
    L = D-AdjM
    Lrw = sp.diag(1/np.sum(AdjM,axis=0)) @ L
    #Lrw = np.diag(np.sum(AdjM,axis=1) ** (-0.5)) @ L @ np.diag(np.sum(AdjM,axis=1) ** (-0.5))
    #L = np.zeros((N, N))
    #for ii in range(N):
    #    L[ii, :] = AdjM[ii, :]/np.sum(AdjM[ii, :])
    #    L[ii, ii] = -1

    d_Lrw, V_Lrw = la.eig(Lrw)
    #(d_Lrw, V_Lrw) = sp.sparse.linalg.eigs(Lrw, k=evs, which="SM")
    # print(d_L)
    return Lrw, d_Lrw, V_Lrw


def graph_laplacian_eigs_sparse(AdjM, N, evs):
    D_inv = sp.sparse.diags(1 / AdjM.sum(axis=1).A.ravel())
    L = D_inv @ AdjM - sp.sparse.eye(N)
    (d_L, V_L) = sp.sparse.linalg.eigs(L, evs, which="SM")
    return L, d_L, V_L

"""
def generate_time_sequece(L, N, T, h, m, dt, v=0):
    # initialize time series
    u = np.zeros((N, T + 1))
    # initialize u(0) and set u(1)=u(0)
    for ii in range(N):
        u[ii, 0] = np.random.rand(1, 1)
        u[ii, 1] = u[ii, 0] + v * np.random.rand(1, 1)
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
    return u
"""

def generate_time_sequece(L, N, h, m, V, K, M, dt, v=0):
    # initialize time series
    T = K + M
    u = np.zeros((N, T)) + np.zeros((N, T)) * np.array([0.+1.j])
    # initialize u(0) and set u(1)=u(0)
    for ii in range(N):
        u[ii, 0] = np.array([np.random.rand()+np.random.rand()*1.j])#np.random.rand(1, 1)
    # df_u = pd.DataFrame(u[:, 0], columns=["u"])
    # df_u.to_csv("u_init.csv", index=False)
    imag = np.array([0.+1.j])
    V = V#(h / dt) * (0.5 - (h * dt / m))- imag * (h / (2 * dt))
    R_tilde = - imag  * ((dt * h) / (2 * m)) * L + (1 - imag * (dt * V) / h) * np.eye(N)
    
    w, v = la.eig(L)
    w1, v1 = la.eig(R_tilde)
    w1 = -(2*m)/(dt*h) * np.imag(w1)
    
    # generate time sequence for each node
    #u1 = u.copy()
    for tt in range(T-1):
        u[:, tt + 1] = R_tilde @ u[:, tt]
        #u[:, tt + 1] = - imag  * ((dt * h) / (2 * m)) * L @ u[:, tt] + u[:, tt]
    #dmd = DMD(svd_rank=2)
    #dmd.fit(u.T)
    #(d, V) = dmd(u[:,:-1], u[:,1:], svThresh=1.e-5)
    #la.norm(V[:,1].real-v[:,1])
    #max(la.norm(u,axis=0))
    #plt.plot(v[:,1], label='Ground Truth')
    #plt.plot(V[:,1].real, label='DMD')
    #plt.axhline(y=0, linestyle=':', color='k')
    #plt.legend()
    return u, R_tilde


def append_time_sequence(u, L, T):
    (N, T0) = u.shape
    u = np.hstack((u, np.zeros((N, T))))

    c = np.sqrt(2.) - 1.e-6

    # generate time sequence for each node
    for tt in range(T0 - 1, T0 + T - 1):
        # for kk in range(N):
        #     tmp = 0
        #     for ll in range(N):
        #         if L[kk, ll]:
        #             tmp += L[kk, ll] * u[ll, tt]
        #     u[kk, tt + 1] = 2. * u[kk, tt] - u[kk, tt - 1] + c ** 2 * tmp
        u[:, tt + 1] = 2. * u[:, tt] - u[:, tt - 1] + c ** 2 * L @ u[:, tt]
    return u


def graph_clustering(L, d_L, N, T, v_inds, Nrows, case, base):
    n_inds = len(v_inds)
    Ncols = T + 1 - Nrows

    # initialize time series
    u = generate_time_sequece(L, N, T)

    # generate input matrices
    XX = np.zeros((Nrows, Ncols))

    dmdcoeff = np.zeros((N, n_inds))
    for i in range(N):
        # DMD
        for jj in range(Nrows):
            XX[jj, :] = u[i, jj:jj + Ncols]

        X = XX[:, :-1]
        Y = XX[:, 1:]
        (d, V) = dmd(X, Y, svThresh=1.e-5)

        if i == 0:
            # printVector(d.real, f'Node{i + 1}_d_{case}')
            eigenvalues = np.append(np.reshape(d.real, (1, -1)), np.reshape(d.imag, (1, -1)), axis=0)
            df_real = pd.DataFrame(eigenvalues, index=["real", "imag"])
            df_real.to_csv(f"./eigenvalues_{case}.csv")

            # plot eigen modes
            plt.figure()
            plt.plot(np.real(V[:, 0]), '-o')
            plt.plot(np.real(V[:, 1]), '-s')
            plt.plot(np.real(V[:, 2]), 'v-.')
            plt.plot(np.real(V[:, 3]), '^:')
            plt.plot(np.real(V[:, 4]), '*--')
            plt.tight_layout()
            plt.savefig(f"./output_{base}/modes_real_{case}_node{i + 1}.png")
            plt.close()

            plt.figure()
            plt.plot(np.imag(V[:, 0]), '-o')
            plt.plot(np.imag(V[:, 1]), '-s')
            plt.plot(np.imag(V[:, 2]), 'v-.')
            plt.plot(np.imag(V[:, 3]), '^:')
            plt.plot(np.imag(V[:, 4]), '*--')
            plt.tight_layout()
            plt.xlabel("k")
            plt.savefig(f"./output_{base}/modes_imag_{case}_node{i + 1}.png")
            plt.close()

            # plot eigenvalues
            plt.figure()
            plt.plot(np.real(d_L), np.imag(d_L), 's', alpha=0.8, label='exact')
            plt.plot(np.real(d), np.imag(d), 'o', alpha=0.8, label='DMD')
            plt.xlabel('Real')
            plt.ylabel('Imag')
            plt.legend()
            plt.xlim([-1.2, 1.2])
            plt.ylim([-1.2, 1.2])
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(f"./output_{base}/eigenvalues_{case}_node{i + 1}.png")
            plt.close()

        # compute coefficients proportional to kth Laplacian eigenvector
        for k_ind, v_ind in enumerate(v_inds):
            v_ind2 = v_ind * 2 - 1
            dmdcoeff[i, k_ind] = np.real(sp.linalg.lstsq(V, X[:, 0])[0][v_ind2] * V[0, v_ind2])
            freq = np.angle(d_L[v_ind2 + 1])
            # print(f"Freq {k_ind}: {freq}")

    return dmdcoeff


def graph_clustering_static(u, R_tilde, L, d_L, h, m, V, dt, K, M, v_inds, case, base, node_num=[2]):
    n_inds = len(v_inds)
    (N, T) = u.shape
    dominant_freqs = np.zeros((N, n_inds))

    # generate input matrices
    XX = np.zeros((K, M+1)) + np.zeros((K, M+1)) * np.array([0.+1.j])

    dmdcoeff = np.zeros((N, n_inds)) + np.zeros((N, n_inds)) * np.array([0.+1.j])
    for i in range(N):
        # DMD
        for jj in range(M+1):
            XX[:, jj] = u[i, jj:jj + K]

        X = XX[:, :-1]
        Y = XX[:, 1:]
        
        #X1 = np.zeros((N, N)) + np.zeros((N, N)) * np.array([0.+1.j])
        #Y1 = np.zeros((N, N)) + np.zeros((N, N)) * np.array([0.+1.j])
        #for jj in range(N):
        #    X1[:, jj] = u[i,jj:jj+N]
        #    Y1[:, jj] = u[i,jj+1:jj+N+1]
        
        (d, V) = dmd(X, Y, svThresh=1e-6)
        d = d ** (1 / dt)
        eval_L_dmd = - (2 * m) / (dt * h) * d.imag
        eval_L = eval_L_dmd
        # print(f"Node {i} DMD Freq: {freq}")
        #eval_L, evec_L = la.eig(L)
        #eval_R, evec_R = la.eig(R_tilde)
        
        #eval_R_L = -(2 * m)/(dt * h) * np.imag(eval_R)
        
        #Phi = np.zeros((K,K)) + np.array([0.+1.j]) * np.zeros((K,K))
        #for i in range(K):
        #    Phi[i,:] = np.power(eval_R,i)
        
        #A = Y @ la.pinv(X)
        #A_phi = Phi @ np.diag(eval_R) @ la.pinv(Phi)
        #error = np.max(abs(A@Phi-Phi@np.diag(eval_R))/abs(Phi@np.diag(eval_R)))
        #error1 = np.max(abs(A-A_phi)/abs(A))
        #d1, _ = la.eig(A)
        #eval_L1 = - (2 * m) / (dt * h) * d1.imag

        if i in node_num:
            # printVector(d.real, f'Node{i + 1}_d_{case}')
            #eigenvalues = np.append(np.reshape(d.real, (1, -1)), np.reshape(d.imag, (1, -1)), axis=0)
            #df_real = pd.DataFrame(eigenvalues, index=["real", "imag"])
            #df_real.to_csv(f"/Users/xingzixu/clustering/canonical/schr/result/output_{base}/eigenvalues_{case}.csv")

            # plot eigen modes
            plt.figure()
            plt.plot(np.real(V[:, 0]), '-o')
            plt.plot(np.real(V[:, 1]), '-s')
            plt.plot(np.real(V[:, 2]), 'v-.')
            #plt.plot(np.real(V[:, 3]), '^:')
            #plt.plot(np.real(V[:, 4]), '*--')
            plt.xlabel("k")
            plt.tight_layout()
            plt.savefig(f"/Users/xingzixu/clustering/canonical/schr/figure/plots_{base}/modes_real_{case}_node{i + 1}.png")
            plt.close()

            plt.figure()
            plt.plot(np.imag(V[:, 0]), '-o')
            plt.plot(np.imag(V[:, 1]), '-s')
            plt.plot(np.imag(V[:, 2]), 'v-.')
            #plt.plot(np.imag(V[:, 3]), '^:')
            #plt.plot(np.imag(V[:, 4]), '*--')
            plt.xlabel("k")
            plt.tight_layout()
            plt.savefig(f"/Users/xingzixu/clustering/canonical/schr/figure/plots_{base}/modes_imag_{case}_node{i + 1}.png")
            plt.close()

            # plot eigenvalues
            plt.figure()
            plt.plot(np.real(d_L), np.imag(d_L), 's', alpha=0.8, label='exact')
            plt.plot(np.real(eval_L), np.imag(eval_L), 'o', alpha=0.8, label='DMD')
            plt.xlabel('Real')
            plt.ylabel('Imag')
            plt.legend()
            plt.xlim([-1.2, 1.2])
            plt.ylim([-1.2, 1.2])
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(f"/Users/xingzixu/clustering/canonical/schr/figure/plots_{base}/eigenvalues_{case}_node{i + 1}.png")
            plt.close()

        # compute coefficients proportional to kth Laplacian eigenvector
        for k_ind, v_ind in enumerate(v_inds):
            # v_ind2 = v_ind * 2 - 1
            v_ind0 = np.where(abs(eval_L) > min(abs(eval_L)))[0]
            v_ind1 = np.argmin(abs(eval_L[v_ind0]))
            v_ind2 = v_ind0[v_ind1]
            dmdcoeff[i, k_ind] = sp.linalg.lstsq(V, X[:, 0])[0][v_ind2]
            dominant_freqs[i, k_ind] = np.angle(d[v_ind2])

    return dmdcoeff, dominant_freqs


def graph_clustering_dynamic(u, pre_dom_freqs, d_M, v_inds, Nrows, case, base, node_num=0):
    n_inds = len(v_inds)
    (N, T) = u.shape
    Ncols = T + 1 - Nrows
    dominant_freqs = np.zeros((N, n_inds))

    # generate input matrices
    XX = np.zeros((Nrows, Ncols))

    dmdcoeff = np.zeros((N, n_inds))
    for i in range(N):
        # DMD
        for jj in range(Nrows):
            XX[jj, :] = u[i, jj:jj + Ncols]

        X = XX[:, :-1]
        Y = XX[:, 1:]
        (d, V) = dmd(X, Y, svThresh=1.e-5)
        # freq = np.angle(d[:5])
        # print(f"Node {i} DMD Freq: {freq}")

        if i in node_num:
            # printVector(d.real, f'Node{i + 1}_d_{case}')
            eigenvalues = np.append(np.reshape(d.real, (1, -1)), np.reshape(d.imag, (1, -1)), axis=0)
            df_real = pd.DataFrame(eigenvalues, index=["real", "imag"])
            df_real.to_csv(f"/Users/xingzixu/clustering/canonical/schr/result/output_{base}/eigenvalues_{case}.csv")

            # plot eigen modes
            plt.figure()
            plt.plot(np.real(V[:, 0]), '-o')
            plt.plot(np.real(V[:, 1]), '-s')
            plt.plot(np.real(V[:, 2]), 'v-.')
            plt.plot(np.real(V[:, 3]), '^:')
            plt.plot(np.real(V[:, 4]), '*--')
            plt.tight_layout()
            plt.savefig(f"/Users/xingzixu/clustering/canonical/schr/figure/plots_{base}/modes_real_{case}_node{i + 1}.png")
            plt.close()

            plt.figure()
            plt.plot(np.imag(V[:, 0]), '-o')
            plt.plot(np.imag(V[:, 1]), '-s')
            plt.plot(np.imag(V[:, 2]), 'v-.')
            plt.plot(np.imag(V[:, 3]), '^:')
            plt.plot(np.imag(V[:, 4]), '*--')
            plt.tight_layout()
            plt.xlabel("k")
            plt.savefig(f"/Users/xingzixu/clustering/canonical/schr/figure/plots_{base}/modes_imag_{case}_node{i + 1}.png")
            plt.close()

            # plot eigenvalues
            plt.figure()
            plt.plot(np.real(d_M), np.imag(d_M), 's', alpha=0.8, label='exact')
            plt.plot(np.real(d), np.imag(d), 'o', alpha=0.8, label='DMD')
            plt.xlabel('Real')
            plt.ylabel('Imag')
            plt.legend()
            plt.xlim([-1.2, 1.2])
            plt.ylim([-1.2, 1.2])
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(f"/Users/xingzixu/clustering/canonical/schr/figure/plots_{base}/eigenvalues_{case}_node{i + 1}.png")
            plt.close()

        # compute coefficients proportional to kth Laplacian eigenvector
        for k_ind, v_ind in enumerate(v_inds):
            # v_ind2 = v_ind * 2 - 1
            v_ind0 = np.where(np.angle(d) > 1.e-6)[0]
            v_ind1 = np.argmin(np.abs(np.angle(d[v_ind0]) - pre_dom_freqs[i, k_ind]))
            v_ind2 = v_ind0[v_ind1]
            dmdcoeff[i, k_ind] = np.real(sp.linalg.lstsq(V, X[:, 0])[0][v_ind2] * V[0, v_ind2])
            dominant_freqs[i, k_ind] = np.angle(d[v_ind2])
            print(f"Node {i} DMD Freq: {dominant_freqs[i, k_ind]}, Previous: {pre_dom_freqs[i, k_ind]}")

    return dmdcoeff, dominant_freqs