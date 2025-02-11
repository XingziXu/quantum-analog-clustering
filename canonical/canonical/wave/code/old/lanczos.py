import numpy as np

def householder_qr(A):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    for i in range(n):
        H_i = make_householder(R[i:, i])
        H = np.eye(m)
        H[i:, i:] = H_i
        R = H @ R
        Q = Q @ H
    return Q, R

def make_householder(a):
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H

def qr_algorithm(A, num_iters=10):
    S = A.copy()
    n = S.shape[0]
    
    for _ in range(num_iters):
        Q, R = householder_qr(S)
        S = R @ Q

    eigenvalues = np.diag(S)
    return eigenvalues

# Example usage:
N=100
A = np.random.rand(N,N)
ATA = A.T @ A
eval, _ = np.linalg.eig(ATA)
_, s, _ = np.linalg.svd(A)

abs(qr_algorithm(ATA)-eval)/(abs(eval))
print(qr_algorithm(ATA))
