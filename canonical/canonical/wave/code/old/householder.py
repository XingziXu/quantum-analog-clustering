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

def commutator(A, B):
    return A @ B - B @ A

def skew_symmetric(A):
    #print(np.tril(A,-1) - np.tril(A,-1).T)
    return np.tril(A,-1) - np.tril(A,-1).T

#def toda_step(y, t):
#    dydt = commutator(y, skew_symmetric(y))
#    return dydt

def toda_flow(A, T, dt):
    #print(int(np.round(T/dt)))
    #print(np.linalg.norm(A))
    for i in range(0, int(np.round(T/dt))):
        A = A + dt * commutator(A, skew_symmetric(A))
        #print(np.linalg.norm(A))
        #print(commutator(A, skew_symmetric(A)))
    return A

def householder(A):
    N = A.shape[0]
    alpha = -np.sign(A[1,0])*np.sqrt(A[1:,0].T @ A[1:,0])
    r = np.sqrt(0.5 * (alpha ** 2 - A[1,0] * alpha))
    v = A[:,0] / (2 * r)
    v[0] = 0
    v[1] = (A[1,0]-alpha) / (2 * r)
    v = v.reshape(v.shape[0],1)
    P = np.eye(N) - 2 * v @ v.T
    A = P @ A @ P
    
    for k in reversed(range(2, N-1)):
        alpha = -np.sign(A[k,k-1])*np.sqrt(A[k:,k-1].T @ A[k:,k-1])
        r = np.sqrt(0.5 * (alpha ** 2 - A[k,k-1] * alpha))
        v = A[:,k-1] / (2 * r)
        v[:k] = 0
        v[k] = (A[k,k-1] - alpha) / (2 * r)
        v = v.reshape(v.shape[0],1)
        P = np.eye(N) - 2 * v @ v.T
        A = P @ A @ P
    return A
    
np.random.seed(1)
N=100
A = np.random.rand(N,N)
ATA = A.T @ A
tri_A = householder(ATA)

eval, _ = np.linalg.eig(ATA)
_, s, _ = np.linalg.svd(A)

diff = np.zeros((1,N))
Tmax = 2

for T in range(1,Tmax):
    tri_A_toda = toda_flow(A=ATA, T=T*1e-3, dt=5e-4)
    #tri_A_toda = odeintw(func=toda_step, y0=tri_A, t=np.array([0.,T]))
    s3 = np.diag(tri_A_toda)
    diff = np.append(diff, np.reshape(np.sqrt(abs(s3))-s, (1,N)), axis=0)
    #diff = np.append(diff, np.reshape(s3-eval, (1,N)), axis=0)
    #plt.plot(np.sqrt(abs(diff[T,:]))/s)
    #plt.plot(np.sign(s3))
    #plt.plot(np.sign(np.real(eval)))
    #plt.plot(abs(diff[T,:]/np.real(eval)), color=lighten_color('b', T/Tmax))
    plt.plot(diff[T,:]/s * 100, color=lighten_color('b', T/Tmax))
    plt.xlabel('Node number')
    plt.ylabel('Error Percentage')

plt.show()


print(s)
print(np.diag(tri_A_toda))