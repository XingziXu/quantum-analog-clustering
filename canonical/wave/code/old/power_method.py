import numpy as np
import math
import matplotlib.pyplot as plt
#from scipy.integrate import odeint
import odeintw

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def power_iteration(A, num_iterations: int):
    # Randomly initialize a vector
    b_k = np.random.rand(A.shape[1],1)

    for _ in range(num_iterations):
        # calculate the matrix-by-vector product Ab
        b_k1 = A @ b_k

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k


def rayleigh_quotient_iteration(A, num_iterations: int):
    n = A.shape[0]
    I = np.eye(n)

    # Placeholder for eigenvalues and eigenvectors
    eigenvalues = np.empty(n)
    eigenvectors = np.empty((n, n))
    
    for k in range(n):
        # Start with a random guess
        x = np.random.rand(n)
        x /= np.linalg.norm(x)

        for _ in range(num_iterations):
            # Compute Rayleigh quotient
            mu = np.dot(x.T, np.dot(A, x)) / np.dot(x.T, x)

            # Calculate inverse of the shifted matrix
            try:
                B = np.linalg.inv(A - mu * I)
            except np.linalg.LinAlgError:
                # The matrix is singular, use a small shift instead
                B = np.linalg.inv(A - (mu + 1e-6) * I)

            # Normalize the vector
            x = np.dot(B, x)
            x /= np.linalg.norm(x)

        # Store the eigenvalue and eigenvector
        eigenvalues[k] = mu
        eigenvectors[:, k] = x

        # Deflate the matrix
        A -= mu * np.outer(x, x)

    return eigenvalues, eigenvectors

np.random.seed(1)
N=10
A = np.random.rand(N,N)
ATA = A.T @ A

s1, _ = np.linalg.eig(ATA)
s1 = np.sqrt(s1)
_, s2, _ = np.linalg.svd(A)

diff = np.zeros((1,N))
Kmax = 100

s3 = np.zeros((Kmax, N))

s3, _ = rayleigh_quotient_iteration(A=ATA, num_iterations=100)

for k in reversed(range(0,Kmax)):
    bk = power_iteration(A=ATA, num_iterations=k)
    eval_0 = ((bk.T @ ATA @ bk) / (bk.T @ bk)).item()
    s3[k,0] = eval_0
    ATA = ATA - np.eye(N) * eval_0
    for n in range(1,N):
        bk = power_iteration(A=ATA, num_iterations=k)
        eval_1 = ((bk.T @ ATA @ bk) / (bk.T @ bk)).item()
        eval_0 = eval_0 + eval_1
        s3[k,n] = eval_0
        
        ATA = ATA - np.eye(N) * eval_0
plt.show()


print(s)
print(np.diag(tri_A_toda))