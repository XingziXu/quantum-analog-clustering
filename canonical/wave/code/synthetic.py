import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import os
from sklearn.cluster import KMeans
from utils import degree_vector, generate_time_sequence_qiskit, graph_clustering_static

G = nx.Graph()
#G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), 
                  (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), 
                  (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), 
                  (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), 
                  (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), 
                  (6, 7), (6, 8), (6, 9), (6, 10), 
                  (7, 8), (7, 9), (7, 10), 
                  (8, 9), (8, 10), 
                  (9, 10),
                  (11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (11, 19), (11, 20),
                  (12, 13), (12, 14), (12, 15), (12, 16), (12, 17), (12, 18), (12, 19), (12, 20), 
                  (13, 14), (13, 15), (13, 16), (13, 17), (13, 18), (13, 19), (13, 20), 
                  (14, 15), (14, 16), (14, 17), (14, 18), (14, 19), (14, 20), 
                  (15, 16), (15, 17), (15, 18), (15, 19), (15, 20), 
                  (16, 17), (16, 18), (16, 19), (16, 20), 
                  (17, 18), (17, 19), (17, 20), 
                  (18, 19), (18, 20), 
                  (19, 20),                  
                  (21, 22), (21, 23), (21, 24), (21, 25), (21, 26), (21, 27), (21, 28), (21, 29), (21, 30),
                  (22, 23), (22, 24), (22, 25), (22, 26), (22, 27), (22, 28), (22, 29), (22, 30), 
                  (23, 24), (23, 25), (23, 26), (23, 27), (23, 28), (23, 29), (23, 30), 
                  (24, 25), (24, 26), (24, 27), (24, 28), (24, 29), (24, 30), 
                  (25, 26), (25, 27), (25, 28), (25, 29), (25, 30), 
                  (26, 27), (26, 28), (26, 29), (26, 30), 
                  (27, 28), (27, 29), (27, 30), 
                  (28, 29), (28, 30), 
                  (29, 30),
                  (31, 32), (31, 33), (31, 34), (31, 35), (31, 36), (31, 37), (31, 38), (31, 39), (31, 40),
                  (32, 33), (32, 34), (32, 35), (32, 36), (32, 37), (32, 38), (32, 39), (32, 40), 
                  (33, 34), (33, 35), (33, 36), (33, 37), (33, 38), (33, 39), (33, 40), 
                  (34, 35), (34, 36), (34, 37), (34, 38), (34, 39), (34, 40), 
                  (35, 36), (35, 37), (35, 38), (35, 39), (35, 40), 
                  (36, 37), (36, 38), (36, 39), (36, 40), 
                  (37, 38), (37, 39), (37, 40), 
                  (38, 39), (38, 40), 
                  (39, 40),
                  (1, 11), (11, 21), (21, 31), (31, 1)])

A=[]
for ed in G.edges():
    A.append([ed[0], ed[1]])
A = np.array(A)

num_node = G.number_of_nodes()
evs = 3
v_inds = range(1, evs)
Nrows = num_node
T = Nrows * 2 - 1
num_itr = 500
deflation = "schur"
base = "synthetic"
case = f"DMD_synthetic_T{T}_rows{Nrows}"
start = 0
power = True
base_dir=os.getcwd()

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
rand_v = np.random.rand(num_node)
u_qiskit = generate_time_sequence_qiskit(AdjM=AdjM, IncMt=IncMt, N=num_node, T=T, rand_v = rand_v)
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
                                                    start=start, 
                                                    power=power)

"""
kmeans = KMeans(n_clusters=4, random_state=0, init='k-means++', n_init=10, max_iter=300, tol=1e-4, algorithm='lloyd')
kmeans.fit(np.reshape(dmdcoeff[:,0], (40,1)))
labels_pred = kmeans.labels_
centers_pred = kmeans.cluster_centers_

kmeans = KMeans(n_clusters=4, random_state=0, init='random', n_init=10, max_iter=300, tol=1e-4, algorithm='lloyd')
kmeans.fit(np.reshape(V_L_B[:,1], (40,1)))
labels_gt = kmeans.labels_
centers_gt = kmeans.cluster_centers_
#accuracy = np.maximum(np.sum(np.sign(dmdcoeff[:,0]) == np.sign(V_L_B[:,1])), np.sum(np.sign(dmdcoeff[:,0]) == -np.sign(V_L_B[:,1])))/num_node * 100.
#print(accuracy)
"""

plt.plot(dmdcoeff[:,0]/la.norm(dmdcoeff[:,0]), label='DMD',color='blue')
plt.plot(V_L_B[:,1]/la.norm(V_L_B[:,1]), label='Ground Truth',color='orange')
plt.legend()
plt.savefig(base_dir+'/canonical/wave/figure/sym_synthetic.pdf')
plt.clf()
plt.close()

pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes
options = {"edgecolors": "tab:gray", "node_size": 100, "alpha": 1.0}
nx.draw_networkx_nodes(G, pos, nodelist=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], node_color="tab:red", **options)
nx.draw_networkx_nodes(G, pos, nodelist=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20], node_color="tab:green", **options)
nx.draw_networkx_nodes(G, pos, nodelist=[21, 22, 23, 24, 25, 26, 27, 28, 29, 30], node_color="tab:blue", **options)
nx.draw_networkx_nodes(G, pos, nodelist=[31, 32, 33, 34, 35, 36, 37, 38, 39, 40], node_color="tab:orange", **options)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=[(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), 
                  (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), 
                  (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10), 
                  (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), 
                  (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), 
                  (6, 7), (6, 8), (6, 9), (6, 10), 
                  (7, 8), (7, 9), (7, 10), 
                  (8, 9), (8, 10), 
                  (9, 10)],
    width=1,
    alpha=0.9,
    edge_color="black",
)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=[(11, 12), (11, 13), (11, 14), (11, 15), (11, 16), (11, 17), (11, 18), (11, 19), (11, 20),
                  (12, 13), (12, 14), (12, 15), (12, 16), (12, 17), (12, 18), (12, 19), (12, 20), 
                  (13, 14), (13, 15), (13, 16), (13, 17), (13, 18), (13, 19), (13, 20), 
                  (14, 15), (14, 16), (14, 17), (14, 18), (14, 19), (14, 20), 
                  (15, 16), (15, 17), (15, 18), (15, 19), (15, 20), 
                  (16, 17), (16, 18), (16, 19), (16, 20), 
                  (17, 18), (17, 19), (17, 20), 
                  (18, 19), (18, 20), 
                  (19, 20)],
    width=1,
    alpha=0.9,
    edge_color="black",
)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=[(21, 22), (21, 23), (21, 24), (21, 25), (21, 26), (21, 27), (21, 28), (21, 29), (21, 30),
                  (22, 23), (22, 24), (22, 25), (22, 26), (22, 27), (22, 28), (22, 29), (22, 30), 
                  (23, 24), (23, 25), (23, 26), (23, 27), (23, 28), (23, 29), (23, 30), 
                  (24, 25), (24, 26), (24, 27), (24, 28), (24, 29), (24, 30), 
                  (25, 26), (25, 27), (25, 28), (25, 29), (25, 30), 
                  (26, 27), (26, 28), (26, 29), (26, 30), 
                  (27, 28), (27, 29), (27, 30), 
                  (28, 29), (28, 30), 
                  (29, 30)],
    width=1,
    alpha=0.9,
    edge_color="black",
)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=[(31, 32), (31, 33), (31, 34), (31, 35), (31, 36), (31, 37), (31, 38), (31, 39), (31, 40),
                  (32, 33), (32, 34), (32, 35), (32, 36), (32, 37), (32, 38), (32, 39), (32, 40), 
                  (33, 34), (33, 35), (33, 36), (33, 37), (33, 38), (33, 39), (33, 40), 
                  (34, 35), (34, 36), (34, 37), (34, 38), (34, 39), (34, 40), 
                  (35, 36), (35, 37), (35, 38), (35, 39), (35, 40), 
                  (36, 37), (36, 38), (36, 39), (36, 40), 
                  (37, 38), (37, 39), (37, 40), 
                  (38, 39), (38, 40), 
                  (39, 40)],
    width=1,
    alpha=0.9,
    edge_color="black",
)
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=[(1, 11), (11, 21), (21, 31), (31, 1)],
    width=1,
    alpha=0.9,
    edge_color="black",
)
plt.show()
plt.savefig("/scratch/xx84/clustering/canonical/wave/figure/synthetic.pdf")
plt.clf()
plt.close()