import numpy as np
import numpy.linalg as la
import scipy as sp
from qiskit_dynamics.solvers import Solver
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

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

def dmd(X, Y, num_itr=500, mode='exact', svThresh=1.e-10, which="LM", deflation='schur', power=True):
    '''
    Exact and standard DMD of the data matrices X and Y.

    :param mode: 'exact' for exact DMD or 'standard' for standard DMD
    :return:     eigenvalues d and modes Phi
    '''
    if power:
        s, Vt = svd_power(X=X, num_itr=num_itr, deflation=deflation) # the cols of V are the corresponding eigenvectors, since XTX=XXT, we havce V=U.T
    else:
        _, s, Vt = np.linalg.svd(X, full_matrices=True)
        s = np.square(s)
    
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
    try:
        U = X @ la.pinv(Vt) @ np.diag(1/s)
    except:
        U = X @ la.pinv(np.nan_to_num(Vt)) @ np.diag(1/s)
        print("There were NaN or inf in Vt, filtered them out first")
    S_inv = np.diag(1/s)
    A = U.conj().T @ Y @ Vt.conj().T @ S_inv
    # d, W = sortEig(A, A.shape[0])
    # d, W = sortEig(A, r)
    n = A.shape[0]
    if r < n:
        d, W = sp.sparse.linalg.eigs(A, r, which=which)
    else:
        try:
            d, V = sp.linalg.eig(A)
        except:
            d, V = sp.linalg.eig(np.nan_to_num(A))
            print("There were NaN or inf in A, filtered them out first")
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
    #L = np.zeros((N, N))
    #for ii in range(N):
    #    L[ii, :] = AdjM[ii, :]/np.sum(AdjM[ii, :])
    #    L[ii, ii] = -1
    L = np.eye(N) - np.diag(np.power(degree_vector(AdjM), -1)) @ AdjM
    
    # eigen system of static graph
    I = np.eye(N=N)
    # assemble M = [I * 2 + 2*L, -I; I, 0*I]
    M = np.concatenate((np.concatenate((I * 2 + 2 * L, -I), axis=1), np.concatenate((I, 0 * I), axis=1)), axis=0)
    (d_M, V_M) = sp.sparse.linalg.eigs(M, M.shape[0], which="LM")
    (d_L, V_L) = sp.sparse.linalg.eigs(L, evs, which="SM")
    # print(d_L)
    return L, d_M, V_M, d_L, V_L

def generate_time_sequence(L, N, T, rand_v, dt, v=0):
    # initialize time series
    u = np.zeros((N, T + 1))
    # initialize u(0) and set u(1)=u(0)
    
    #for ii in range(N):
    #    u[ii, 0] = np.random.rand(1, 1)
    #    u[ii, 1] = u[ii, 0] + v * np.random.rand(1, 1)
    
    # df_u = pd.DataFrame(u[:, 0], columns=["u"])
    # df_u.to_csv("u_init.csv", index=False)
    u[:,0] = rand_v
    u[:,1] = rand_v
    c = np.sqrt(2.) - 1.e-6

    # generate time sequence for each node
    for tt in range(1, T):
        # for kk in range(N):
        #     tmp = 0
        #     for ll in range(N):
        #         if L[kk, ll]:
        #             tmp += L[kk, ll] * u[ll, tt]
        #     u[kk, tt+1] = 2.*u[kk, tt] - u[kk, tt-1] + c**2*tmp
        u[:, tt + 1] = 2. * u[:, tt] - u[:, tt - 1] - c ** 2 * L @ u[:, tt]
    return u

def generate_time_sequence_B(B, T, rand_v, dt):
    nn, ne = B.shape
    c = np.sqrt(1.) - 1.e-6
    u_B = np.zeros((nn+ne, T + 1)) + 1.j * np.zeros((nn+ne, T + 1))
    rand_v = rand_v#np.random.rand(nn)
    rand_e = np.zeros(ne,)#-1.j * c * (B.T @ rand_v)
    u_B[:, 0] = np.concatenate((rand_v, rand_e))
    v_update = np.concatenate((np.concatenate((np.zeros((nn,nn)), B),axis=1), np.concatenate((B.T, np.zeros((ne,ne))),axis=1)), axis=0)
    a_update = np.concatenate((np.concatenate((B @ B.T, np.zeros((nn,ne))),axis=1), np.concatenate((np.zeros((ne,nn)), B.T @ B),axis=1)), axis=0)
    v_B = np.zeros_like(u_B)
    v_B[:, 0] = np.zeros_like(u_B[:, 0]) - (dt/2) * (c ** 2) * a_update @ u_B[:, 0]
    for ii in range(T-1):
        u_B[:, ii+1] = u_B[:, ii] + dt * v_B[:,ii]
        v_B[:, ii+1] = v_B[:, ii] - dt * (c ** 2) * a_update @ u_B[:, ii+1]
    return np.real(u_B[:nn, :])

#def time_symm_1(B, lamb):
#    nn = B.shape[0]
#    ne = B.shape[1]
#    c = np.sqrt(2.) - 1.e-6
#    update = - c ** 2 * np.concatenate((np.concatenate(((B @ B.T), np.zeros((nn, ne))), axis=1), np.concatenate((np.zeros((ne, nn)), (B.T @ B)), axis=1)), axis=0)
    

def generate_time_sequence_qiskit(AdjM, IncMt, N, T, rand_v, v=0):
    # First calculate B matrix properly
    c = np.sqrt(2.) - 1.e-6
    eps = 1e-12
    D_nsqrt = np.diag(np.power(degree_vector(AdjM) + eps, -0.5))
    B = D_nsqrt @ IncMt / np.sqrt(2.)
    nn, ne = B.shape

    # calculate ground truth eigen value and eigen vector
    #L_B = B @ B.T
    #d_L_B, V_L_B = la.eig(L_B)

    # Create Hermitian drift operator with proper structure
    drift = 1j * c * np.block([
        [np.zeros((nn, nn)), B],
        [-B.conj().T, np.zeros((ne, ne))]
    ])

    # Verify Hermiticity
    print("Is drift Hermitian?", np.allclose(drift, drift.conj().T))
    #initial = np.random.rand(B.shape[0])
    #u_B_0 = np.concatenate((initial,np.zeros(B.shape[1])))

    rand_v = np.random.rand(nn)
    rand_e = np.zeros(ne,)#-1.j * c * (B.T @ rand_v)
    u_B_0 = np.concatenate((rand_v, rand_e))
    
    u_B = np.zeros((N, T))
    u_B[:,0] = u_B_0[:N]
    #u_B = np.zeros((N, T+1)) + np.imag(np.zeros((N, T+1)))
    # First, check if drift matrix is correct
    print("B matrix first few elements:")
    print(B[:5,:5])
    print("\nDrift matrix upper right block:")
    print(drift[:5,nn:nn+5])
    print("\nDrift matrix lower left block:")
    print(drift[nn:nn+5,:5])

    hamiltonian_solver = Solver(
        static_hamiltonian=drift,  # Now drift is already Hermitian
        validate=True
    )

    # After setting up drift matrix
    t_evolution = np.linspace(0, T-1, T, dtype=np.float64)

    # Store initial condition
    u_B[:,0] = u_B_0[:N]

    # Solve entire evolution at once with more fine-grained control
    sol = hamiltonian_solver.solve(
        t_span=[0, T-1],  # Single time span for entire evolution
        y0=u_B_0,
        t_eval=t_evolution,  # Evaluate at specific times
        atol=1e-14,
        rtol=1e-14,
        method='RK45',
        max_dt=0.1  # Control step size
    )

    # Extract solutions
    for i in range(1, T):
        u_B[:,i] = sol.y[i,:N]
        
        # Debug first few steps
        if i < 3:
            print(f"Time step {i}, first few values:", u_B[:5,i])
            print(f"State diff from previous:", np.linalg.norm(u_B[:,i] - u_B[:,i-1]))
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

def graph_clustering_static(u, d_M, v_inds, num_itr, deflation, Nrows, case, base, base_dir, power, gnd_truth, start, node_num=[0]):
    n_inds = len(v_inds)
    (N, T) = u.shape
    Ncols = T + 1 - Nrows
    dominant_freqs = np.zeros((N, n_inds))
    
    # generate input matrices
    XX = np.zeros((Nrows, Ncols))
    dmdcoeff = np.zeros((N, n_inds))
    
    for i in tqdm(range(start, N), desc="Calculating clusters with DMD..."):
        # DMD
        for jj in range(Nrows):
            XX[jj, :] = u[i, jj:jj + Ncols]

        X = XX[:, :-1]
        Y = XX[:, 1:]
        (d, V) = dmd(X, Y, svThresh=1.e-5, num_itr=num_itr, deflation=deflation, power=power)
        freq = np.angle(d[:5])

        if i in node_num:
            # [Previous plotting code remains the same]
            pass

        # compute coefficients proportional to kth Laplacian eigenvector
        for k_ind, v_ind in enumerate(v_inds):
            # Find eigenvalues with angle > threshold
            v_ind0 = np.where(np.angle(d) > 1.e-6)[0]
            
            if len(v_ind0) == 0:
                # Handle case where no eigenvalues meet the criteria
                print(f"Warning: No eigenvalues with angle > 1.e-6 found for node {i}, k_ind {k_ind}")
                # Use the eigenvalue with the largest positive angle instead
                angles = np.angle(d)
                if np.any(angles > 0):
                    v_ind2 = np.argmax(angles)
                else:
                    # If no positive angles, use the first eigenvalue
                    v_ind2 = 0
            else:
                v_ind1 = np.argmin(np.angle(d[v_ind0]))
                v_ind2 = v_ind0[v_ind1]
            
            # Calculate coefficients
            dmdcoeff[i, k_ind] = np.real(sp.linalg.lstsq(V, X[:, 0])[0][v_ind2] * V[0, v_ind2])
            dominant_freqs[i, k_ind] = np.angle(d[v_ind2])
            
            correct_so_far = np.maximum(
                np.sum(np.sign(dmdcoeff[:i+1,k_ind]) == np.sign(gnd_truth[:i+1])),
                np.sum(np.sign(dmdcoeff[:i+1,k_ind]) == -np.sign(gnd_truth[:i+1]))
            )
        print(f' We have got {correct_so_far}/{i-start+1} correct')
        
        # Save coefficients
        with open('.\\canonical\\wave\\result\\output_twitter\\dyn_qiskit_dmdcoeff.npy', 'wb') as f:
            np.save(f, dmdcoeff)
            
    return dmdcoeff, dominant_freqs