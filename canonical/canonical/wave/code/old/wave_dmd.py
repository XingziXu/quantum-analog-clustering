#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.io
import scipy as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt

# import d3s.algorithms as algorithms

import pandas as pd


def dmd(X, Y, mode='exact', svThresh=1.e-10, which="LM"):
    '''
    Exact and standard DMD of the data matrices X and Y.

    :param mode: 'exact' for exact DMD or 'standard' for standard DMD
    :return:     eigenvalues d and modes Phi
    '''
    U, s, Vt = sp.linalg.svd(X, full_matrices=False)
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


def graph_laplacian_eigs(AdjM, N, evs):
    L = np.zeros((N, N))
    for ii in range(N):
        L[ii, :] = AdjM[ii, :]/np.sum(AdjM[ii, :])
        L[ii, ii] = -1

    # eigen system of static graph
    I = np.eye(N=N)
    # assemble M = [I * 2 + 2*L, -I; I, 0*I]
    M = np.concatenate((np.concatenate((I * 2 + 2 * L, -I), axis=1), np.concatenate((I, 0 * I), axis=1)), axis=0)
    (d_M, V_M) = sp.sparse.linalg.eigs(M, M.shape[0], which="LM")
    (d_L, V_L) = sp.sparse.linalg.eigs(L, evs, which="SM")
    # print(d_L)
    return L, d_M, V_M, d_L, V_L


def graph_laplacian_eigs_sparse(AdjM, N, evs):
    D_inv = sp.sparse.diags(1 / AdjM.sum(axis=1).A.ravel())
    L = D_inv @ AdjM - sp.sparse.eye(N)
    I = sp.sparse.eye(N=N)
    M = sp.sparse.vstack(sp.sparse.hstack(I * 2 + 2 * L, -I), sp.sparse.hstack(I, 0 * I))
    (d_M, V_M) = sp.sparse.linalg.eigs(M, N * 2, which="LM")
    (d_L, V_L) = sp.sparse.linalg.eigs(L, evs, which="SM")
    return L, d_M, V_M, d_L, V_L


def generate_time_sequece(L, N, T, v=0):
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


def graph_clustering(L, d_M, N, T, v_inds, Nrows, case, base):
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
            plt.plot(np.real(d_M), np.imag(d_M), 's', alpha=0.8, label='exact')
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
            freq = np.angle(d_M[v_ind2 + 1])
            # print(f"Freq {k_ind}: {freq}")

    return dmdcoeff


def graph_clustering_static(u, d_M, v_inds, Nrows, case, base, node_num=[0]):
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
        
        XTX = X.T @ X
        svd_toda = toda_flow(A=XTX, T=0.1, dt=0.01)
        U, S, Vh = np.linalg.svd(XTX)
        
        (d, V) = dmd(X, Y, svThresh=1.e-5)
        freq = np.angle(d[:5])
        # print(f"Node {i} DMD Freq: {freq}")

        if i in node_num:
            # printVector(d.real, f'Node{i + 1}_d_{case}')
            eigenvalues = np.append(np.reshape(d.real, (1, -1)), np.reshape(d.imag, (1, -1)), axis=0)
            df_real = pd.DataFrame(eigenvalues, index=["real", "imag"])
            df_real.to_csv(f"./output_{base}/eigenvalues_{case}.csv")

            # plot eigen modes
            plt.figure()
            plt.plot(np.real(V[:, 0]), '-o')
            plt.plot(np.real(V[:, 1]), '-s')
            plt.plot(np.real(V[:, 2]), 'v-.')
            plt.plot(np.real(V[:, 3]), '^:')
            plt.plot(np.real(V[:, 4]), '*--')
            plt.xlabel("k")
            plt.tight_layout()
            plt.savefig(f"./plots_{base}/modes_real_{case}_node{i + 1}.png")
            plt.close()

            plt.figure()
            plt.plot(np.imag(V[:, 0]), '-o')
            plt.plot(np.imag(V[:, 1]), '-s')
            plt.plot(np.imag(V[:, 2]), 'v-.')
            plt.plot(np.imag(V[:, 3]), '^:')
            plt.plot(np.imag(V[:, 4]), '*--')
            plt.xlabel("k")
            plt.tight_layout()
            plt.savefig(f"./plots_{base}/modes_imag_{case}_node{i + 1}.png")
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
            plt.savefig(f"./plots_{base}/eigenvalues_{case}_node{i + 1}.png")
            plt.close()

        # compute coefficients proportional to kth Laplacian eigenvector
        for k_ind, v_ind in enumerate(v_inds):
            # v_ind2 = v_ind * 2 - 1
            v_ind0 = np.where(np.angle(d) > 1.e-6)[0]
            v_ind1 = np.argmin(np.angle(d[v_ind0]))
            v_ind2 = v_ind0[v_ind1]
            dmdcoeff[i, k_ind] = np.real(sp.linalg.lstsq(V, X[:, 0])[0][v_ind2] * V[0, v_ind2])
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
            df_real.to_csv(f"./output_{base}/eigenvalues_{case}.csv")

            # plot eigen modes
            plt.figure()
            plt.plot(np.real(V[:, 0]), '-o')
            plt.plot(np.real(V[:, 1]), '-s')
            plt.plot(np.real(V[:, 2]), 'v-.')
            plt.plot(np.real(V[:, 3]), '^:')
            plt.plot(np.real(V[:, 4]), '*--')
            plt.tight_layout()
            plt.savefig(f"./plots_{base}/modes_real_{case}_node{i + 1}.png")
            plt.close()

            plt.figure()
            plt.plot(np.imag(V[:, 0]), '-o')
            plt.plot(np.imag(V[:, 1]), '-s')
            plt.plot(np.imag(V[:, 2]), 'v-.')
            plt.plot(np.imag(V[:, 3]), '^:')
            plt.plot(np.imag(V[:, 4]), '*--')
            plt.tight_layout()
            plt.xlabel("k")
            plt.savefig(f"./plots_{base}/modes_imag_{case}_node{i + 1}.png")
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
            plt.savefig(f"./plots_{base}/eigenvalues_{case}_node{i + 1}.png")
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

def commutator(A, B):
    return A @ B - B @ A

def skew_symmetric(A):
    #print(np.tril(A,-1) - np.tril(A,-1).T)
    return np.tril(A,-1) - np.tril(A,-1).T

def toda_flow(A, T, dt):
    #print(int(np.round(T/dt)))
    print(np.linalg.norm(A))
    for i in range(0, int(np.round(T/dt))):
        A = A + dt * commutator(A, skew_symmetric(A))
        print(np.linalg.norm(A))
        #print(commutator(A, skew_symmetric(A)))
    return A