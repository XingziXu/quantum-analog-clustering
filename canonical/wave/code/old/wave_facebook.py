import qiskit
from packaging import version
from qiskit_dynamics.solvers import Solver
import numpy as np
import pandas as pd
import scipy.io
import scipy as sp
import matplotlib.pyplot as plt
import math
import numpy.linalg as la
import qiskit_dynamics
from scipy.optimize import leastsq
import networkx as nx
import os.path

from qiskit import BasicAer
backend = BasicAer.get_backend('qasm_simulator')


def power_iteration(A, num_iterations: int):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1])
    for _ in range(num_iterations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)
        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)
        # re normalize the vector
        b_k = b_k1 / b_k1_norm
    return b_k

def svd_power(X, num_itr):
    A = X.T @ X
    N = A.shape[0]
    mu = np.zeros(N)
    V = np.zeros_like(A)
    
    for i in range(0, N):
        bk = power_iteration(A, num_itr)
        bk = bk.reshape(bk.shape[0],1)
        muk = ((bk.T @ A @ bk) / (bk.T @ bk)).squeeze()
        mu[i] = muk
        #print(muk)
        tempA = A-muk*np.eye(N)
        #print(tempA)
        b = tempA[:, 0].copy()
        vk = np.linalg.lstsq(tempA[:, 1:], -b)[0]
        vk = np.r_[1, vk]
        #print(vk)
        vk /= np.linalg.norm(vk)
        V[:,i] = vk
        vk = vk.reshape(N,1)
        A = A - vk @ vk.T @ A @ vk @ vk.T
    return mu, V.T

def dmd(X, Y, num_itr, mode='exact', svThresh=1.e-10, which="LM"):
    '''
    Exact and standard DMD of the data matrices X and Y.

    :param mode: 'exact' for exact DMD or 'standard' for standard DMD
    :return:     eigenvalues d and modes Phi
    '''
    
    #s, Vt = svd_power(X=X, num_itr=num_itr) # the cols of V are the corresponding eigenvectors, since XTX=XXT, we havce V=U.T
    
    _, s, Vt = np.linalg.svd(X, full_matrices=True)
    s = np.square(s)
    
    #signs = np.repeat(np.expand_dims(np.sign(Vt1[:,0]),-1),Vt1.shape[1],axis=1)
    #Vt = np.multiply(Vt,signs)
    
    #print(s)
    
    s[s<0] = 0
    s = np.sqrt(s)
    #U = Vt.T
    
    #U, s, Vt = sp.linalg.svd(X, full_matrices=False)
    #mask_U = np.random.random(U.shape[0]) >= 0.5
    #mask_V = np.random.random(U.shape[0]) >= 0.5
    #U[:,mask_U] = U[:,mask_U] * (-1)
    #Vt[mask_V,:] = Vt[mask_V,:] * (-1)
    #Vt = U.T
    
    # remove zeros from s
    s = s[s > svThresh]
    r = len(s)
    #U = U[:, :r]
    Vt = Vt[:r, :]
    U = X @ la.pinv(Vt) @ np.diag(1/s)
    S_inv = np.diag(1/s)
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
        Phi = Y @ Vt.T @ S_inv @ W @ np.diag(1/d)
    elif mode == 'standard':
        Phi = U @ W
    elif mode == 'eigenvector':
        Phi = W
    else:
        raise ValueError('Only exact and standard DMD available.')

    return d, Phi

def degree_vector(AdjM):
    D = np.sum(AdjM, axis=1)
    return D

def graph_laplacian_eigs_sym(AdjM, N, evs):
    D_nsqrt = np.diag(np.power(degree_vector(AdjM), -0.5))
    L = np.eye(N) - D_nsqrt @ AdjM @ D_nsqrt
    #L = -L
    # eigen system of static graph
    I = np.eye(N=N)
    # assemble M = [I * 2 + 2*L, -I; I, 0*I]
    M = np.concatenate((np.concatenate((I * 2 + 2 * L, -I), axis=1), np.concatenate((I, 0 * I), axis=1)), axis=0)
    (d_M, V_M) = sp.sparse.linalg.eigs(M, M.shape[0], which="LM")
    (d_L, V_L) = sp.sparse.linalg.eigs(L, evs, which="SM")
    # print(d_L)
    return L, d_M, V_M, d_L, V_L

def graph_laplacian_eigs(AdjM, N, evs):
    D = degree_vector(AdjM)
    L = np.eye(N) - np.diag(1/D) @ AdjM
    L = -L
    #L = np.zeros((N, N))
    #for ii in range(N):
    #    L[ii, :] = AdjM[ii, :]/np.sum(AdjM[ii, :])
    #    L[ii, ii] = -1

    #print(L-L1)
    # eigen system of static graph
    I = np.eye(N=N)
    # assemble M = [I * 2 + 2*L, -I; I, 0*I]
    M = np.concatenate((np.concatenate((I * 2 + 2 * L, -I), axis=1), np.concatenate((I, 0 * I), axis=1)), axis=0)
    (d_M, V_M) = sp.sparse.linalg.eigs(M, M.shape[0], which="LM")
    (d_L, V_L) = sp.sparse.linalg.eigs(L, evs, which="SM")
    # print(d_L)
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
    c = np.sqrt(1.) - 1.e-6

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

def wave_dyn(T, N, solver, B):
    #Nrows = N
    #Ncols = N+1
    #T = Nrows + Ncols

    # Neumann initial condition
    u0 = np.concatenate((np.random.rand(B.shape[0]),np.zeros(B.shape[1])))
    ut = np.zeros((N, T),dtype=np.complex_)
    ut[:,0] = u0[:N]

    for i in range(1, T):
        t_span = np.array([0., i])
        ut[:,i] = solver.solve(t_span=t_span, y0=u0).y[-1,:N]
    return ut

def graph_clustering_static(u, d_L, v_inds, Nrows, case, base, node_num=[0]):
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
        (d, V) = dmd(X, Y, svThresh=1.e-6, num_itr=5000)
        freq = np.angle(d[:5])
        #print(i)
        #print(d)
        # print(f"Node {i} DMD Freq: {freq}")

        if i in node_num:
            # printVector(d.real, f'Node{i + 1}_d_{case}')
            eigenvalues = np.append(np.reshape(d.real, (1, -1)), np.reshape(d.imag, (1, -1)), axis=0)
            df_real = pd.DataFrame(eigenvalues, index=["real", "imag"])
            df_real.to_csv(f"/Users/xingzixu/clustering/canonical/wave/result/output_{base}/eigenvalues_{case}.csv")

            # plot eigen modes
            plt.figure()
            plt.plot(np.real(V[:, 0]), '-o')
            plt.plot(np.real(V[:, 1]), '-s')
            plt.plot(np.real(V[:, 2]), 'v-.')
            plt.plot(np.real(V[:, 3]), '^:')
            plt.plot(np.real(V[:, 4]), '*--')
            plt.xlabel("k")
            plt.tight_layout()
            plt.savefig(f"/Users/xingzixu/clustering/canonical/wave/figure/plots_{base}/modes_real_{case}_node{i + 1}.png")
            plt.close()

            plt.figure()
            plt.plot(np.imag(V[:, 0]), '-o')
            plt.plot(np.imag(V[:, 1]), '-s')
            plt.plot(np.imag(V[:, 2]), 'v-.')
            plt.plot(np.imag(V[:, 3]), '^:')
            plt.plot(np.imag(V[:, 4]), '*--')
            plt.xlabel("k")
            plt.tight_layout()
            plt.savefig(f"/Users/xingzixu/clustering/canonical/wave/figure/plots_{base}/modes_imag_{case}_node{i + 1}.png")
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
            plt.savefig(f"/Users/xingzixu/clustering/canonical/wave/figure/plots_{base}/eigenvalues_{case}_node{i + 1}.png")
            plt.close()

        # compute coefficients proportional to kth Laplacian eigenvector
        for k_ind, v_ind in enumerate(v_inds):
            # v_ind2 = v_ind * 2 - 1
            #print(d)
            v_ind0 = np.where(np.angle(d) > 1.e-7)[0]
            if len(v_ind0) > 0:
                v_ind1 = np.argmin(np.angle(d[v_ind0]))
                v_ind2 = v_ind0[v_ind1]
                dmdcoeff[i, k_ind] = np.real(sp.linalg.lstsq(V, X[:, 0])[0][v_ind2] * V[0, v_ind2])
                dominant_freqs[i, k_ind] = np.angle(d[v_ind2])
            else:
                dmdcoeff[i, k_ind] = np.real(sp.linalg.lstsq(V, X[:, 0])[0][0] * V[0, 0])
                dominant_freqs[i, k_ind] = np.angle(d[0])
    return dmdcoeff, dominant_freqs

np.random.seed(1234)
plt.rcParams.update({"font.size": 28, "lines.linewidth": 3})

# load graph data
edge = pd.read_csv(r'/Users/xingzixu/clustering/canonical/schr/data/facebook_combined.txt', sep=' ')
edge = np.asarray(edge)
G = nx.Graph()
for idx,row in enumerate(edge):
    #edge[idx,:] = np.asarray(np.char.split(row.astype(np.str_))[0]).astype(np.int_)
    G.add_edge(edge[idx,0], edge[idx,1])

# number of nodes we are concerned with
#max_num_node = np.max(edge)
num_node = 30 #np.max(edge)
#node_numbers = np.random.choice(range(max_num_node), num_node, replace=False)

evs = 3
v_inds = range(1, evs)
T = num_node * 2
Nrows = int(T / 2 + 1)
base = "facebook"
case = f"DMD_karate_T{T}_rows{Nrows}"
#print(case)

# Adjacent matrix and Laplacian for static graph
G1 = nx.Graph()
all_edges=np.array([[0,0]])
for i in range(1, num_node):
    node_edges = np.array([e for e in G.edges(i)])
    node_edges = node_edges[node_edges[:,1]<num_node,:]
    all_edges = np.concatenate((all_edges, np.asarray(node_edges)),axis=0)
    for node_edge in node_edges:
        G1.add_edge(node_edge[0], node_edge[1])

all_edges = all_edges[1:,:]
AdjM = np.asarray(nx.adjacency_matrix(G1).todense())
N = AdjM.shape[0]

# Incidence matrix and Laplacian for static graph

IncMt = np.zeros((N, all_edges.shape[0]))
# AdjMt = np.zeros((N, N))
for ii in range(0, all_edges.shape[0]):
    edge_now = all_edges[ii,:]
    IncMt[edge_now[0]-1,ii] = -1
    IncMt[edge_now[1]-1,ii] = 1
#IncMt = np.concatenate((IncMt, 2.j * np.diag(np.sqrt(degree_vector(AdjM)))), axis=1)
#IncMt = np.concatenate((IncMt, np.eye(N)), axis=1)

#L, d_M, V_M, d_L, V_L = graph_laplacian_eigs(AdjM=AdjM, N=N, evs=evs)
#L_sym, d_M_sym, V_M_sym, d_L_sym, V_L_sym = graph_laplacian_eigs_sym(AdjM=AdjM, N=N, evs=evs)
#L, d_M, V_M, d_L, V_L = graph_laplacian_eigs_sym(AdjM=AdjM, N=N, evs=evs)
#D_nsqrt = np.diag(np.power(degree_vector(AdjM), -0.5))
#L = D_nsqrt @ IncMt @ IncMt.T @ D_nsqrt

# calculate the B matrix
D_nsqrt = np.diag(np.power(degree_vector(AdjM), -0.5))
B = D_nsqrt @ IncMt / np.sqrt(2.)

# calculate ground truth eigen value and eigen vector
L_B = B @ B.T
d_L_B, V_L_B = la.eig(L_B)

# define propagation speed
c = np.sqrt(2.) - 1.e-6

# form static Hamiltonian matrix
H_static = c * np.concatenate((np.concatenate((np.zeros((B.shape[0],B.shape[0])), B),axis=1),np.concatenate((B.conj().T, np.zeros((B.shape[1],B.shape[1]))),axis=1)), axis=0)

hamiltonian = qiskit_dynamics.models.HamiltonianModel(static_operator=H_static)

# define Schrodinger equation
solver = Solver(static_hamiltonian = H_static, validate = True)

initial = np.random.rand(B.shape[0])

u_B_0 = np.concatenate((initial,np.zeros(B.shape[1])))
u_B = np.zeros((N, T+1)) + np.imag(np.zeros((N, T+1)))
if not(os.path.isfile('/Users/xingzixu/clustering/canonical/schr/data/u_B_facebook_qasm_'+str(T)+'_'+str(num_node)+'.npy')):
    u_B[:,0] = u_B_0[:N]

    for i in range(1, T+1):
        t_span = np.array([0., i])
        u_B[:,i] = solver.solve(t_span=t_span, y0=u_B_0).y[-1,:N]

    with open('/Users/xingzixu/clustering/canonical/schr/data/u_B_facebook_qasm_'+str(T)+'_'+str(num_node)+'.npy', 'wb') as f:
        np.save(f, u_B)

else:
    u_B = np.load('/Users/xingzixu/clustering/canonical/schr/data/u_B_facebook_'+str(T)+'_'+str(num_node)+'.npy', mmap_mode='r')

"""
# initialize time series
u_B1 = np.zeros((N, T + 1))
# initialize u(0) and set u(1)=u(0)
#for ii in range(N):
#    u[ii, 0] = np.random.rand(1, 1)
#    u[ii, 1] = u[ii, 0] + v * np.random.rand(1, 1)
# df_u = pd.DataFrame(u[:, 0], columns=["u"])
# df_u.to_csv("u_init.csv", index=False)
u_B1[:,0] = initial
u_B1[:,1] = initial
c = np.sqrt(2.) - 1.e-6

# generate time sequence for each node
for tt in range(1, T):
    # for kk in range(N):
    #     tmp = 0
    #     for ll in range(N):
    #         if L[kk, ll]:
    #             tmp += L[kk, ll] * u[ll, tt]
    #     u[kk, tt+1] = 2.*u[kk, tt] - u[kk, tt-1] + c**2*tmp
    u_B1[:, tt + 1] = 2. * u_B1[:, tt] - u_B1[:, tt - 1] + c ** 2 * (L_B) @ u_B1[:, tt]
    """
dmdcoeff, dominant_freqs = graph_clustering_static(u=u_B, d_L=d_L_B, v_inds=v_inds, Nrows=Nrows, case=case, base=base, node_num=[2])

accuracy = np.maximum(np.sum(np.sign(dmdcoeff[:,0]) == np.sign(V_L_B[:,1])), np.sum(np.sign(dmdcoeff[:,0]) == -np.sign(V_L_B[:,1])))/N * 100.

plt.plot(np.sign(dmdcoeff[:,0]), label='DMD',color='blue')
plt.plot(np.sign(V_L_B[:,1]), label='Ground Truth',color='orange')
plt.show()
#plt.legend()