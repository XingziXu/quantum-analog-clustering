# Running checks for the correct dependencies
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
from qiskit.quantum_info import Operator
from qiskit.quantum_info.states import Statevector
#import jax
#jax.config.update("jax_enable_x64", True)

# tell JAX we are using CPU
#jax.config.update('jax_platform_name', 'cpu')

# import Array and set default backend
#from qiskit_dynamics.array import Array
#Array.set_default_backend('jax')

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

def svd_power(X, num_itr, deflation='schur'):
    #print(num_itr)
    #print(deflation)
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
        try:
            vk = la.lstsq(tempA[:, 1:], -b)[0]
        except:
            try:
                vk = la.pinv(tempA[:, 1:]) @ (-b)
                print("lstsq failed, used pinv instead")
            except:
                if 'vk' in locals():
                    vk = vk[1:].squeeze()
                    print("both lstsq and pinv failed, used previous vk as placeholder")
                else:
                    vk = np.random.randn(N-1,)
                    print("both lstsq and pinv failed, used random vk as placeholder")
            
        
        vk = np.r_[1, vk]
        #print(vk)
        vk /= np.linalg.norm(vk)
        V[:,i] = vk
        vk = vk.reshape(N,1)
        if deflation == 'schur':
            A = A - (A @ vk @ vk.T @ A)/(vk.T @ A @ vk) # Schur's complement deflation
        elif deflation == 'projection':
            A = (np.eye(A.shape[0]) - vk @ vk.T) @ A @ (np.eye(A.shape[0]) - vk @ vk.T) # projection deflation
        elif deflation == 'hotelling':
            A = A - vk @ vk.T @ A @ vk @ vk.T # Hotelling's deflation
        elif deflation == 'ohd':
            if i == 0: # orthogonized Hotelling's deflation
                qk = vk
                A = A - qk @ qk.T @ A @ qk @ qk.T
                Qk = vk
            else:
                qk = (np.eye(A.shape[0]) - Qk @ Qk.T) @ vk
                qk = qk / la.norm(qk)
                A = A - qk @ qk.T @ A @ qk @ qk.T
                Qk = np.concatenate((Qk,qk), axis=1)
        else:
            raise ValueError('Deflaton methods available: hotelling, projection, schur, ohd')
    return mu, V.T

def dmd(X, Y, num_itr=500, mode='exact', svThresh=1.e-10, which="LM", deflation='schur'):
    '''
    Exact and standard DMD of the data matrices X and Y.

    :param mode: 'exact' for exact DMD or 'standard' for standard DMD
    :return:     eigenvalues d and modes Phi
    '''
    
    s, Vt = svd_power(X=X, num_itr=num_itr, deflation=deflation) # the cols of V are the corresponding eigenvectors, since XTX=XXT, we havce V=U.T
    #_, s, Vt = np.linalg.svd(X, full_matrices=True)
    #s = np.square(s)
    
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
    A = U.conj().T @ Y @ Vt.conj().T @ S_inv
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

def generate_time_sequence_qiskit(AdjM, IncMt, N, T, v=0):
    # calculate the B matrix
    D_nsqrt = np.diag(np.power(degree_vector(AdjM), -0.5))
    B = D_nsqrt @ IncMt / np.sqrt(2.)

    # calculate ground truth eigen value and eigen vector
    L_B = B @ B.T
    d_L_B, V_L_B = la.eig(L_B)

    # define propagation speed
    c = np.sqrt(1.) - 1.e-6


    drift = c * np.concatenate((np.concatenate((np.zeros((B.shape[0],B.shape[0])), B),axis=1),np.concatenate((B.conj().T, np.zeros((B.shape[1],B.shape[1]))),axis=1)), axis=0)

    initial = np.random.rand(B.shape[0])
    """u = np.zeros((drift.shape[0], T+1))
    u[:N,0] = initial
    
    #M = np.eye(drift.shape[0]) - 1j * drift
    for tt in range(0, T):
        # for kk in range(N):
        #     tmp = 0
        #     for ll in range(N):
        #         if L[kk, ll]:
        #             tmp += L[kk, ll] * u[ll, tt]
        #     u[kk, tt+1] = 2.*u[kk, tt] - u[kk, tt-1] + c**2*tmp
        u[:, tt + 1] = u[:, tt] - 1j * drift @ u[:, tt]
    """
    u_B_0 = np.concatenate((initial,np.zeros(B.shape[1])))
    u_B = np.zeros((N, T))
    u_B[:,0] = u_B_0[:N]
    #u_B = np.zeros((N, T+1)) + np.imag(np.zeros((N, T+1)))
    hamiltonian_solver = Solver(static_hamiltonian=drift, validate = True)
    t_spans = []
    for i in range(1, T+1):
        t_spans.append(np.array([0.,i]))
    sol = hamiltonian_solver.solve(
    t_span=t_spans, # time interval to integrate over 
    y0=u_B_0, # initial state 
    #t_eval=times, # points to integrate over
    #method='jax_RK4',
    atol=1e-20,
    rtol=2.5e-14
    )
    for i in range(0, T):
        u_B[:,i] = sol[i].y[-1,:N]

    return u_B

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

def graph_clustering_static(u, d_M, v_inds, num_itr, deflation, Nrows, case, base, node_num=[0]):
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
        (d, V) = dmd(X, Y, svThresh=1.e-5, num_itr=num_itr, deflation=deflation)
        freq = np.angle(d[:5])
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
            plt.plot(np.real(d_M), np.imag(d_M), 's', alpha=0.8, label='exact')
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
            v_ind0 = np.where(np.angle(d) > 1.e-6)[0]
            v_ind1 = np.argmin(np.angle(d[v_ind0]))
            v_ind2 = v_ind0[v_ind1]
            dmdcoeff[i, k_ind] = np.real(sp.linalg.lstsq(V, X[:, 0])[0][v_ind2] * V[0, v_ind2])
            dominant_freqs[i, k_ind] = np.angle(d[v_ind2])

    return dmdcoeff, dominant_freqs

dataset_load = 'twitter'

# load graph data
if dataset_load == 'facebook':
    A = pd.read_csv(r'/Users/xingzixu/clustering/canonical/schr/data/facebook_combined.txt', sep=' ')
    A = np.asarray(A)
    A.sort()
    A = A + 1
elif dataset_load == 'karate':
    data = scipy.io.loadmat('/Users/xingzixu/clustering/canonical/schr/data/karate.mat', squeeze_me=True)
    for s in data.keys():
        if s[:2] == '__' and s[-2:] == '__': continue
        exec('%s = data["%s"]' % (s, s))
    A.sort()
elif dataset_load == 'twitter':
    # the twitter dataset is found at http://snap.stanford.edu/data/congress-twitter.html
    A = pd.read_csv(r'/Users/xingzixu/clustering/canonical/wave/data/congress.edgelist', sep=' ').to_numpy()[:,:2]
    A = np.asarray(A)
    A.sort()
    A = A + 1

#backend = DynamicsBackend(
#    solver=solver, subsystem_dims=[3, 3]
#)

num_node = 80
#num_node = np.max(A[:, :2])
A = A[A[:,0]<=num_node]
A = A[A[:,1]<=num_node]


# Adjacent matrix and Laplacian for static graph
N = np.max(A[:, :2])
AdjMt0 = np.zeros((N, N))
# AdjMt = np.zeros((N, N))
L0 = np.zeros((N, N))
for ii in range(A.shape[0]):
    AdjMt0[A[ii, 0] - 1][A[ii, 1] - 1] = 1
AdjM = (AdjMt0 + AdjMt0.T) / 2.

np.random.seed(1234)
plt.rcParams.update({"font.size": 28, "lines.linewidth": 3})

evs = 3
v_inds = range(1, evs)
Nrows = num_node
T = Nrows * 2 - 1
#T = N
#Nrows = int((T + 1) / 2)
base = "karate"
case = f"DMD_karate_T{T}_rows{Nrows}"
deflation = 'schur'
num_itr = 2000
print(case)

# Incidence matrix and Laplacian for static graph

IncMt = np.zeros((N, A.shape[0]))
# AdjMt = np.zeros((N, N))
for ii in range(0, A.shape[0]):
    edge = A[ii,:]
    IncMt[edge[0]-1,ii] = -1
    IncMt[edge[1]-1,ii] = 1

# calculate the B matrix
D_nsqrt = np.diag(np.power(degree_vector(AdjM), -0.5))
B = D_nsqrt @ IncMt / np.sqrt(2.)

# calculate ground truth eigen value and eigen vector
L_B = B @ B.T
d_L_B, V_L_B = la.eig(L_B)


#L, d_M, V_M, d_L, V_L = graph_laplacian_eigs_sym(AdjM=AdjM, N=N, evs=evs)
#u = generate_time_sequece(L=L, N=N, T=T)

u = generate_time_sequence_qiskit(AdjM=AdjM, IncMt=IncMt, N=N, T=T)

dmdcoeff, dominant_freqs = graph_clustering_static(u=u, d_M=d_L_B, num_itr=num_itr, deflation=deflation, v_inds=v_inds, Nrows=Nrows, case=case, base=base, node_num=[2])

accuracy = np.maximum(np.sum(np.sign(dmdcoeff[:,0]) == np.sign(V_L_B[:,1])), np.sum(np.sign(dmdcoeff[:,0]) == -np.sign(V_L_B[:,1])))/N * 100.
print(accuracy)

plt.plot(np.sign(dmdcoeff[:,0]), label='DMD',color='blue')
plt.plot(np.sign(V_L_B[:,1]), label='Ground Truth',color='orange')
plt.legend()