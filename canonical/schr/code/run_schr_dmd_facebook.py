import numpy as np
import pandas as pd
import scipy.io
import scipy as sp
import matplotlib.pyplot as plt
import networkx as nx
from schr_dmd import (graph_laplacian_eigs, generate_time_sequece, append_time_sequence,
                              graph_clustering_static, graph_clustering_dynamic)
import csv
import pandas as pd

if __name__ == "__main__":
    np.random.seed(1234)
    plt.rcParams.update({"font.size": 28, "lines.linewidth": 3})


    # load graph data
<<<<<<< HEAD
    edge_raw = pd.read_csv(r'/Users/xingzixu/clustering/canonical/schr/data/facebook_combined.txt', sep=' ')
=======
    edge_raw = pd.read_csv(r'/Users/xingzixu/clustering/canonical/schr/data/facebook_combined.txt', sep='\n')
>>>>>>> refs/remotes/origin/main
    edge_raw = np.asarray(edge_raw)
    edge = np.zeros((edge_raw.shape[0],2))
    G = nx.Graph()
    for idx,row in enumerate(edge_raw):
        edge[idx,:] = np.asarray(np.char.split(row.astype(np.str_))[0]).astype(np.int_)
        G.add_edge(edge[idx,0], edge[idx,1])

    # Adjacent matrix and Laplacian for static graph
    AdjM = np.asarray(nx.adjacency_matrix(G).todense())
    #N = AdjM.shape[0]
    N = 100
    accuracy = []
<<<<<<< HEAD
    for N in range(100, 500, 50) :
=======
    for N in range(400, 500, 50) :
>>>>>>> refs/remotes/origin/main
        N = AdjM.shape[0]
        AdjM_current = AdjM[:N,:N]
        
        evs = 10
        v_inds = range(1, evs)
        base = "facebook"
        case = f"DMD_facebook"
        #print(case)

        L, d_L, V_L = graph_laplacian_eigs(AdjM=AdjM_current, N=N, evs=evs)
        freq = np.angle(d_L[:3])
        #print(f"Actual Freq 2: {freq}")

        imag = np.array([0.+1.j])
        h = 2.5e-1 # bigger h value results in bigger and faster waves
        m = 1.
        dt = 0.2
        V = 0#- imag * h / dt # -imag * h / dt usually results in rapidly changing waves, and hard to do dmd with
        K = 1 * N
        M = 1 * N
        u, R_tilde = generate_time_sequece(L=L, N=N, h=h, m=m, V=V, dt=dt, K=K, M=M)

        dmdcoeff, dominant_freqs = graph_clustering_static(u=u, R_tilde=R_tilde, L=L, d_L=d_L, h=h, m=m, V=V, dt=dt, K=K, M=M, v_inds=v_inds, case=case, base=base, node_num=[2])

        accuracy = np.append(accuracy, np.minimum(np.sum(-np.sign(dmdcoeff[:,0]) == np.sign(V_L[:,1])), np.sum(np.sign(dmdcoeff[:,0]) == np.sign(V_L[:,1])))/N * 100.)

        fig = plt.figure()

        plt.subplot(2, 1, 1)
        plt.plot(-dmdcoeff[:,0],label='dmd')
        plt.plot(V_L[:,1],label='direct')
        #plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(-np.sign(dmdcoeff[:,0]),label='dmd')
        plt.plot(np.sign(V_L[:,1]),label='direct')
        
        plt.legend()
        plt.savefig('/Users/xingzixu/clustering/canonical/schr/figure/plots_facebook/dmd_results_facebook_'+str(N)+'.png')
        
        print('error is ', np.minimum(np.sum(-np.sign(dmdcoeff[:,0]) == np.sign(V_L[:,1])), np.sum(np.sign(dmdcoeff[:,0]) == np.sign(V_L[:,1])))/N * 100., '%')
        print('max wave value is ',np.sqrt(np.max(u).imag ** 2 + np.max(u).real ** 2))
        print('min wave value is ',np.sqrt(np.min(u).imag ** 2 + np.min(u).real ** 2))
    
    plt.plot(np.arange(50,500,50),accuracy)
    plt.savefig('/Users/xingzixu/clustering/canonical/schr/figure/plots_facebook/dmd_accuracy_facebook.png')
    