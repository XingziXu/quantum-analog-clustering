import numpy as np
import pandas as pd
import scipy.io
import scipy as sp
import matplotlib.pyplot as plt
import networkx as nx
from wave_dmd import (graph_laplacian_eigs, generate_time_sequece, append_time_sequence,
                              graph_clustering_static, graph_clustering_dynamic)
import csv
import pandas as pd

if __name__ == "__main__":
    np.random.seed(1234)
    plt.rcParams.update({"font.size": 28, "lines.linewidth": 3})


    # load graph data
    edge_raw = pd.read_csv(r'/Users/xingzixu/clustering/canonical/schr/data/facebook_combined.txt', sep='\n')
    edge_raw = np.asarray(edge_raw)
    edge = np.zeros((edge_raw.shape[0],2))
    G = nx.Graph()
    for idx,row in enumerate(edge_raw):
        edge[idx,:] = np.asarray(np.char.split(row.astype(np.str_))[0]).astype(np.int_)
        G.add_edge(edge[idx,0], edge[idx,1])

    # Adjacent matrix and Laplacian for static graph
    AdjM = np.asarray(nx.adjacency_matrix(G).todense())
    N = AdjM.shape[0]
    #N = 100
    T = 1000
    Nrows=1000
    accuracy = []
    #for N in range(50, 350, 50) :
    AdjM_current = AdjM[:N,:N]
        
    evs = 10
    v_inds = range(1, evs)
    base = "facebook"
    case = f"DMD_facebook"
        #print(case)

    L, d_M, V_M, d_L, V_L = graph_laplacian_eigs(AdjM=AdjM_current, N=N, evs=evs)
    freq = np.angle(d_L[:3])
        #print(f"Actual Freq 2: {freq}")

    imag = np.array([0.+1.j])
    h = 4e-2 # bigger h value results in bigger and faster waves
    m = .5
    dt = 1.
    V = 0#- imag * h / dt # -imag * h / dt usually results in rapidly changing waves, and hard to do dmd with
    K = 2 * N
    M = 2 * N
    u = generate_time_sequece(L=L, N=N, T=T)

    dmdcoeff, dominant_freqs = graph_clustering_static(u=u, d_M=d_M, v_inds=v_inds, Nrows=Nrows, case=case, base=base, node_num=[2])

    accuracy = np.append(accuracy, np.minimum(np.sum(np.abs(-np.sign(dmdcoeff[:,0]) - np.sign(V_L[:,1]))), np.sum(np.abs(np.sign(dmdcoeff[:,0]) - np.sign(V_L[:,1]))))/N * 100.)

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
        
    print('error is ', np.sum(np.abs(-np.sign(dmdcoeff[:,0]) - np.sign(V_L[:,1])))/N * 100, '%')
    
    