import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos, cosh, tanh
from numpy.linalg import inv, eig, pinv
from scipy.linalg import svd, svdvals
import scipy as sp
from pydmd import DMD
import 


T = 19
u = np.zeros((T,1))
u[0,:] = 1
imag = np.array([0.+1.j])
for t in range(T-1):
    for j in range(10):
        u[t+1,:] = u[t+1,:] + np.exp(imag * np.pi / 10 * j * t) * j / 10

XX = np.zeros((10,11))
for jj in range(10):
    XX[:,jj] = u[jj:jj + 10,:].squeeze()

X = XX[:, :-1]
Y = XX[:, 1:]
A = Y @ la.pinv(X)

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
    A = U.T.conj() @ Y @ Vt.T.conj() @ S_inv
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
        Phi = Y @ Vt.T.conj() @ S_inv @ W @ sp.diag(1/d)
    elif mode == 'standard':
        Phi = U @ W
    elif mode == 'eigenvector':
        Phi = W
    else:
        raise ValueError('Only exact and standard DMD available.')

    return d, Phi

G = nx.Graph()

G.add_edge(1, 0, weight=0.8)
G.add_edge(2, 3, weight=0.3)
G.add_edge(0, 2, weight=0.7)
G.add_edge(4, 5, weight=0.7)
G.add_edge(4, 3, weight=0.9)
G.add_edge(1, 2, weight=0.8)
G.add_edge(3, 5, weight=0.8)

elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

# nodes
nx.draw_networkx_nodes(G, pos, node_size=700)

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
nx.draw_networkx_edges(
    G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
)

# node labels
nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
# edge weight labels
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()

W = nx.to_numpy_array(G, nonedge=0.)
N = W.shape[0] # number of vertices
T = 2 * N - 1 # define terminal time for testing
h = 1. # planck constant
m = 1. #9.11 * 1e-34 # mass of electron
dt = 1.
V = 0.#- np.array([0+1.j]) * h / dt
u0 = np.random.rand(1,N) # initial u values

D = np.diag(np.sum(W,1))
L = np.eye(N) - np.matmul(inv(D), W)
#L_sym = (D**(-0.5)) @ L @ (D**(-0.5))
imag = np.array([0.+1.j])
R_tilde = ((1 - imag * dt * V / h) * np.eye(N) - imag * h * dt / (2 * m) * L)

ut = np.zeros((T,N)) + np.zeros((T,N)) * np.array([0.+1.j])
ut[0,:] = u0
for idx_t in range(1, num_t):
    ut[idx_t,:] = R_tilde @ ut[idx_t-1,:]



a = np.zeros((N,2*N))
clusters = np.zeros(N)
for j in range(0,N):
    mu, Phi = dmd(X[:,:,j], Y[:,:,j])

    a[j,:] = np.linalg.solve(Phi, X[:,0,j])
    
    for idx, val in enumerate(a[j,:]):
        if val > 0:
            clusters[j] = clusters[j] + val * (2 ** idx)
    
    
w, v = np.linalg.eig(L)
v[:,1]