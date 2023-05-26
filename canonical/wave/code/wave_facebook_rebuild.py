# Running checks for the correct dependencies
from utils import graph_laplacian_eigs, generate_time_sequence, graph_clustering_static, degree_vector
import qiskit
from packaging import version
#from qiskit_dynamics.solvers import Solver
import numpy as np
import pandas as pd
import scipy.io
import scipy as sp
import matplotlib.pyplot as plt
import math
import numpy.linalg as la
#import qiskit_dynamics
from scipy.optimize import leastsq
import networkx as nx
#from qiskit.quantum_info import Operator
#from qiskit.quantum_info.states import Statevector
import json
import os
import yaml

if __name__ == '__main__':
    np.random.seed(1234)
    plt.rcParams.update({"font.size": 28, "lines.linewidth": 3})
    
    base_dir=os.getcwd()
    with open(base_dir+"/canonical/wave/code/wave.yaml", "r") as file:
        hyper_param = yaml.safe_load(file)
    #json_file = open(base_dir+"\\canonical\\wave\\code\\wave.json", "r", encoding="utf-8")
    #hyper_param = json.load(json_file)
    #json_file.close()

    num_node = hyper_param['num_node']
    #num_node = np.max(A[:, :2])
    deflation = hyper_param['deflation']
    num_itr = hyper_param['num_itr']
    evs = 3
    v_inds = range(1, evs)
    Nrows = num_node
    T = Nrows * 2 - 1
    base = hyper_param['dataset']
    case = f"DMD_karate_T{T}_rows{Nrows}"
    print(case)
    
    if hyper_param['dataset']=='facebook':
    # load graph data
        #A = pd.read_csv(r'/Users/xingzixu/clustering/canonical/schr/data/facebook_combined.txt', sep=' ')
        A = pd.read_csv(base_dir + '/canonical/wave/data/facebook_combined.txt', sep=' ')
        A = np.asarray(A)
        A = A + 1
        A = A[A[:,0]<=num_node]
        A = A[A[:,1]<=num_node]
    elif hyper_param['dataset']=='twitter':
        # the twitter dataset is found at http://snap.stanford.edu/data/congress-twitter.html
        #A = pd.read_csv(r'/Users/xingzixu/clustering/canonical/wave/data/congress.edgelist', sep=' ').to_numpy()[:,:2]
        A = pd.read_csv(base_dir + "/canonical/wave/data/congress.edgelist", sep=' ').to_numpy()[:,:2]
        A = np.asarray(A)
        A = A + 1
        A = A[A[:,0]<=num_node]
        A = A[A[:,1]<=num_node]
    elif hyper_param['dataset']=='karate':
        data = scipy.io.loadmat(base_dir + "/canonical/wave/data/" + hyper_param['dataset'], squeeze_me=True)
        for s in data.keys():
            if s[:2] == '__' and s[-2:] == '__': continue
            exec('%s = data["%s"]' % (s, s))
        A = A[A[:,0]<=num_node]
        A = A[A[:,1]<=num_node]

    num_node = max(A[:,1])

    if hyper_param['rw']:
        AdjMt0 = np.zeros((num_node, num_node))
        L0 = np.zeros((num_node, num_node))
        for ii in range(A.shape[0]):
            AdjMt0[A[ii, 0] - 1][A[ii, 1] - 1] = 1
        AdjM = (AdjMt0 + AdjMt0.T) / 2.
        L, d_M, V_M, d_L, V_L = graph_laplacian_eigs(AdjM=AdjM, N=num_node, evs=evs)
        u = generate_time_sequence(L=L, N=num_node, T=T)
        dmdcoeff, dominant_freqs = graph_clustering_static(u=u, d_M=d_L, num_itr=num_itr, deflation=deflation, v_inds=v_inds, Nrows=Nrows, case=case, base=base, base_dir=base_dir, node_num=[2])
        accuracy = np.maximum(np.sum(np.sign(dmdcoeff[:,0]) == np.sign(V_L[:,1])), np.sum(np.sign(dmdcoeff[:,0]) == -np.sign(V_L[:,1])))/num_node * 100.
        print(accuracy)
        plt.plot(np.sign(dmdcoeff[:,0]), label='DMD',color='blue')
        plt.plot(np.sign(V_L[:,1]), label='Ground Truth',color='orange')
        plt.legend()
        plt.savefig(base_dir+'/canonical/wave/figure/rw_'+hyper_param['dataset']+'.png')
    else:
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

        # calculate the B matrix
        D_nsqrt = np.diag(np.power(degree_vector(AdjM), -0.5))
        B = D_nsqrt @ IncMt / np.sqrt(2.)#(2.)

        # calculate ground truth eigen value and eigen vector
        L_B = B @ B.T
        d_L_B, V_L_B = la.eig(L_B)
        
        #M = np.concatenate((np.concatenate((2*np.eye(num_node)-((np.sqrt(2)-1e-6)**2)*L_B, np.zeros((num_node, num_node))), axis=1), np.concatenate((np.zeros((num_node, num_node)), -np.eye(num_node)), axis=1)), axis=0)
        
        rand_v = np.random.rand(num_node)
        
        #u = generate_time_sequence(L=L_B, N=num_node, T=T, rand_v = rand_v, dt=hyper_param['dt'])
        #u_B = generate_time_sequence_B(B=B, T=T, rand_v = rand_v, dt=hyper_param['dt'])
        #u_qiskit = generate_time_sequence_qiskit(AdjM=AdjM, IncMt=IncMt, N=num_node, T=T, rand_v = rand_v)

        #u_qiskit = np.load(base_dir+'\\canonical\\wave\\result\\output_twitter\\dyn_qiskit.npy')
        #with open(base_dir + '\\canonical\\wave\\result\\output_'+hyper_param['dataset']+'\\dyn_qiskit.npy', 'wb') as f:
        #    np.save(f, u_qiskit)
        
        if hyper_param['dynamics'] == 'B':
            with open(base_dir + '/canonical/wave/result/output_'+hyper_param['dataset']+'/dyn_B.npy', 'wb') as f:
                np.save(f, u_B)
            print('Used classically simulated quantum dynamics')
            dmdcoeff, dominant_freqs = graph_clustering_static(u=u_B, 
                                                               d_M=d_L_B, 
                                                               num_itr=num_itr, 
                                                               deflation=deflation, 
                                                               v_inds=v_inds, 
                                                               Nrows=Nrows, 
                                                               case=case, 
                                                               base=base, 
                                                               base_dir=base_dir, 
                                                               node_num=[2], 
                                                               gnd_truth=V_L_B[:,1], 
                                                               start=hyper_param['start'], 
                                                               power=hyper_param['power'])
        elif hyper_param['dynamics'] == 'qiskit':
            with open(base_dir + '/canonical/wave/result/output_'+hyper_param['dataset']+'/dyn_qiskit.npy', 'wb') as f:
                np.save(f, u_qiskit)
            print('Used qiskit simulated quantum dynamics')
            dmdcoeff, dominant_freqs = graph_clustering_static(u=u_qiskit, 
                                                               d_M=d_L_B, 
                                                               num_itr=num_itr, 
                                                               deflation=deflation, 
                                                               v_inds=v_inds, 
                                                               Nrows=Nrows, 
                                                               case=case, 
                                                               base=base, 
                                                               base_dir=base_dir, 
                                                               node_num=[2], 
                                                               gnd_truth=V_L_B[:,1], 
                                                               start=hyper_param['start'], 
                                                               power=hyper_param['power'])
        else:
            with open(base_dir + '/canonical/wave/result/output_'+hyper_param['dataset']+'/dyn_wave.npy', 'wb') as f:
                np.save(f, u)
            print('Used classically simulated wave dynamics')
            dmdcoeff, dominant_freqs = graph_clustering_static(u=u, 
                                                               d_M=d_L_B, 
                                                               num_itr=num_itr, 
                                                               deflation=deflation, 
                                                               v_inds=v_inds, 
                                                               Nrows=Nrows, 
                                                               case=case, 
                                                               base=base, 
                                                               base_dir=base_dir, 
                                                               node_num=[2], 
                                                               gnd_truth=V_L_B[:,1], 
                                                               start=hyper_param['start'], 
                                                               power=hyper_param['power'])
        with open(base_dir + '\\canonical\\wave\\result\\output_'+hyper_param['dataset']+'\\dmdcoeff_'+str(hyper_param['dynamics'])+'.npy', 'wb') as f:
            np.save(f, dmdcoeff)


        accuracy = np.maximum(np.sum(np.sign(dmdcoeff[:,0]) == np.sign(V_L_B[:,1])), np.sum(np.sign(dmdcoeff[:,0]) == -np.sign(V_L_B[:,1])))/num_node * 100.
        print(accuracy)

        plt.plot(np.sign(dmdcoeff[:,0]), label='DMD',color='blue')
        plt.plot(np.sign(V_L_B[:,1]), label='Ground Truth',color='orange')
        plt.legend()
        plt.savefig(base_dir+'/canonical/wave/figure/sym_'+hyper_param['dataset']+'.png')