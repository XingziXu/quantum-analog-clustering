from utils import *
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
from tqdm import tqdm
import os
from qiskit_ibm_provider import IBMProvider
#from qiskit.circuit.QuantumCircuit import Hamiltonian
from qiskit.circuit import QuantumCircuit as qc
from qiskit.circuit.library import UnitaryGate, HamiltonianGate
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator, Options
from qiskit import quantum_info, transpile
from qiskit import execute, Aer
from qiskit.visualization import plot_histogram
from qiskit import IBMQ, transpile
from qiskit.tools.monitor import job_monitor
from qiskit.quantum_info import Statevector
from IPython.display import display, Math

def generate_time_sequence_quantum(AdjM, IncMt, N, T, rand_v, v=0):
    # calculate the B matrix
    D_nsqrt = np.diag(np.power(degree_vector(AdjM), -0.5))
    B = D_nsqrt @ IncMt / np.sqrt(2.)
    nn, ne = B.shape
    
    # define propagation speed
    c = np.sqrt(2.) - 1.e-6
    drift = c * np.concatenate((np.concatenate((np.zeros((B.shape[0],B.shape[0])), B),axis=1),np.concatenate((B.conj().T, np.zeros((B.shape[1],B.shape[1]))),axis=1)), axis=0)
    
    rand_v = np.random.rand(nn)
    rand_e = np.zeros(ne,)#-1.j * c * (B.T @ rand_v)
    u_B_0 = np.concatenate((rand_v, rand_e))
    
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
    #method='DOP853',
    atol=1e-10,
    rtol=1e-10
    )
    for i in range(0, T):
        u_B[:,i] = sol[i].y[-1,:N]

    return u_B


# Save account credentials.
IBMProvider.save_account(token="", overwrite=True)
# Load previously saved account credentials.
provider = IBMProvider()

#phi = 2.
#A = np.array([[1,0],[0,1]])

#circuit = qc(2)
#circuit.hamiltonian(operator=A, time=phi, qubits=range(2))


#data = scipy.io.loadmat('/home/jovyan/clustering/karate.mat', squeeze_me=True)
#for s in data.keys():
#    if s[:2] == '__' and s[-2:] == '__': continue
#    exec('%s = data["%s"]' % (s, s))
A = np.array([[1,2],[1,3],[1,4],[2,3],[2,4],[3,4],[4,7],[5,6],[5,7],[5,8],[6,7],[6,8],[7,8]])
#A = np.array([[1,2],[1,3],[2,3],[3,4]])
num_node = max(A[:,1]).item()
Nrows = int(num_node * 1.0)
T = Nrows * 2 - 1


AdjMt0 = np.zeros((num_node, num_node))
L0 = np.zeros((num_node, num_node))
for ii in range(A.shape[0]):
    AdjMt0[A[ii, 0] - 1][A[ii, 1] - 1] = 1
AdjM = (AdjMt0 + AdjMt0.T) / 2.
IncMt = np.zeros((num_node, A.shape[0]))
for ii in range(0, A.shape[0]):
    edge = A[ii,:]
    IncMt[edge[0]-1,ii] = -1
    IncMt[edge[1]-1,ii] = 1

    
"""
L, d_M, V_M, d_L, V_L = graph_laplacian_eigs(AdjM=AdjM, N=num_node, evs=3)
u = generate_time_sequence(L=L, N=num_node, T=T)
dmdcoeff, dominant_freqs = graph_clustering_static(u=u, d_M=d_L_B, num_itr=2000, deflation="schur", v_inds=range(1, 3), Nrows=Nrows, case=f"DMD_karate_T{T}_rows{Nrows}", base='line', base_dir=os.getcwd(), node_num=[2])
accuracy = np.maximum(np.sum(np.sign(dmdcoeff[:,0]) == np.sign(V_L[:,1])), np.sum(np.sign(dmdcoeff[:,0]) == -np.sign(V_L[:,1])))/num_node * 100.    

"""
# calculate the B matrix
D_nsqrt = np.diag(np.power(degree_vector(AdjM), -0.5))
B = D_nsqrt @ IncMt / np.sqrt(2.)#(2.)

# calculate ground truth eigen value and eigen vector
L_B = B @ B.T
c = np.sqrt(2.) - 1.e-6
drift = c * np.concatenate((np.concatenate((np.zeros((B.shape[0],B.shape[0])), B),axis=1),np.concatenate((B.conj().T, np.zeros((B.shape[1],B.shape[1]))),axis=1)), axis=0)
drift_all = np.eye(32)
drift_all[:drift.shape[0], :drift.shape[1]] = drift
drift = drift_all
d_L_B, V_L_B = la.eig(L_B)

np.random.seed(7)
rand_v = np.random.rand(32)#num_node)
rand_v[21:] = 0.
rand_v = rand_v/la.norm(rand_v)


u_quantum = np.zeros((8, T)) + 1.j * np.zeros((8, T))

for phi in range(1, T+1):
#phi = 1.#np.linspace(1, T, T)
#c = qiskit.ClassicalRegister(3)
    qr = qiskit.QuantumRegister(5)
    #cr = qiskit.ClassicalRegister(3)
    circuit = qc(qr)
    #qc.unitary(L_B,q)
    #circuit = qc(num_node)
    #ham_gate = HamiltonianGate(data=L_B, time=np.linspace(1, T, T), label="ibmq_qasm_simulator")
    #circuit.hamiltonian(operator=Operator(L_B[:, :]), time=phi, qubits=circuit.qubits, label="ibm_nairobi")
    circuit.hamiltonian(operator=Operator(drift), time=phi, qubits=circuit.qubits, label="qasm_simulator")
    #circuit.hamiltonian(operator=Operator(L_B[:, :]), time=phi, qubits=circuit.qubits, label="ibmq_qasm_simulator")
    #circuit.draw('mpl')
    circuit.initialize(rand_v)
    #psi = Statevector(circuit)
    #print(psi)
    #psi.draw('latex')
    #circuit.measure(qr, cr)
    circuit.save_statevector()
    # Choose a backend (simulator in this case)
    backend = Aer.get_backend('qasm_simulator')
    #backend = Aer.get_backend('statevector_simulator')
    #backend = provider.get_backend("ibmq_qasm_simulator")#"ibm_nairobi")

    # Execute the circuit
    # The number of shots is the number of times the circuit is run
    #transpiled_circuit = transpile(circuit, backend)
    #job = execute(circuit.decompose(reps=30), backend, shots=1024)
    job = backend.run(circuit.decompose(reps=10))#execute(circuit, backend)
    #job = execute(experiments=circuit, backend=backend, shots=1024, memory=True)
    #print(Statevector(circuit))
    #job_monitor(job)
    
    #statevector = job.result().get_statevector().data
        
    # Retrieve the results
    u_quantum[:, phi-1] = job.result().get_statevector().data[:8]

# Get the statevector
#counts = result.get_counts(circuit)

#plot_histogram(counts)

"""
options = Options()
#options.optimization_level = 2
#options.resilience_level = 2
observable = quantum_info.SparsePauliOp(["II", "XX", "YY", "ZZ"], coeffs=[1, 1, -1, 1])
#observable = quantum_info.Statevector(data=[1.])#quantum_info.Operator(data=np.eye(L_B.shape[0]))
service = QiskitRuntimeService()
backend = service.backend("ibmq_qasm_simulator")
estimator = Estimator(backend, options=options)
job = estimator.run(circuit, observable)
result = job.result()
"""
#M = np.concatenate((np.concatenate((2*np.eye(num_node)-((np.sqrt(2)-1e-6)**2)*L_B, np.zeros((num_node, num_node))), axis=1), np.concatenate((np.zeros((num_node, num_node)), -np.eye(num_node)), axis=1)), axis=0)



#u = generate_time_sequence(L=L_B, N=num_node, T=T, rand_v = rand_v, dt=1.)
#u_B = generate_time_sequence_B(B=B, T=T, rand_v = rand_v, dt=1.)
#u_qiskit = generate_time_sequence_qiskit(AdjM=AdjM, IncMt=IncMt, N=num_node, T=T, rand_v = rand_v)

dmdcoeff, dominant_freqs = graph_clustering_static(u=u_quantum, 
                                                    d_M=d_L_B, 
                                                    num_itr=100, 
                                                    deflation='schur', 
                                                    v_inds=range(1, 4), 
                                                    Nrows=Nrows, 
                                                    case=f"DMD_line", 
                                                    base='line', 
                                                    base_dir=os.getcwd(), 
                                                    node_num=[2], 
                                                    gnd_truth=V_L_B[:,1], 
                                                    start=0, 
                                                    power=True)
accuracy = np.maximum(np.sum(np.sign(dmdcoeff[:,0]) == np.sign(V_L_B[:,1])), np.sum(np.sign(dmdcoeff[:,0]) == -np.sign(V_L_B[:,1])))/num_node * 100.

print(accuracy)

plt.plot(np.sign(dmdcoeff[:,0]), label='DMD',color='blue')
plt.plot(-np.sign(V_L_B[:,1]), label='Ground Truth',color='orange')
plt.legend()
