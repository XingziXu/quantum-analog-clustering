import numpy as np
import pandas as pd
import scipy.io
import scipy as sp
import matplotlib.pyplot as plt
from graph_clustering import (graph_laplacian_eigs, generate_time_sequece, append_time_sequence,
                              graph_clustering_static, graph_clustering_dynamic)

if __name__ == "__main__":
    np.random.seed(1234)
    plt.rcParams.update({"font.size": 28, "lines.linewidth": 3})

    evs = 3
    v_inds = range(1, evs)
    T = 49
    Nrows = 25
    base = "karate"
    case = f"DMD_karate_T{T}_rows{Nrows}"
    print(case)

    # load graph data
    data = scipy.io.loadmat('../karate.mat', squeeze_me=True)
    for s in data.keys():
        if s[:2] == '__' and s[-2:] == '__': continue
        exec('%s = data["%s"]' % (s, s))

    # Adjacent matrix and Laplacian for static graph
    N = np.max(A[:, :2])
    AdjMt0 = np.zeros((N, N))
    # AdjMt = np.zeros((N, N))
    L0 = np.zeros((N, N))
    for ii in range(A.shape[0]):
        AdjMt0[A[ii, 0] - 1][A[ii, 1] - 1] = 1
    AdjM = (AdjMt0 + AdjMt0.T) / 2.

    L, d_M, V_M, d_L, V_L = graph_laplacian_eigs(AdjM=AdjM, N=N, evs=evs)
    freq = np.angle(d_M[:3])
    print(f"Actual Freq 2: {freq}")

    u = generate_time_sequece(L=L, N=N, T=T)

    dmdcoeff, dominant_freqs = graph_clustering_static(u=u, d_M=d_M, v_inds=v_inds, Nrows=Nrows, case=case, base=base, node_num=[2])

    df_output = pd.DataFrame(V_L.real)
    for k_ind, v_ind in enumerate(v_inds):
        df_output[f"DMD_{v_ind}"] = dmdcoeff[:, k_ind]
        dmd_correct = np.sum((df_output[v_ind].values * df_output[f"DMD_{v_ind}"].values > 0).astype(int))
        print(f"DMD vs Spectral v{v_ind + 1}: {dmd_correct}")

    df_output.to_csv(f"./output_{base}/clustering_{case}.csv")

    for k_ind, v_ind in enumerate(v_inds):
        plt.figure()
        plt.plot(np.arange(1, N + 1), -dmdcoeff[:, k_ind], '-o')
        plt.axhline(y=0, linestyle=':', color='k')
        # plt.xticks(np.arange(1, N+1, 5), np.arange(1, N+1, 5))
        # plt.xticks([1, 100, 200, 300, 400], [1, 100, 200, 300, 400])
        plt.xlabel("Node number")
        plt.ylabel("DMD Coefficient")
        plt.tight_layout()
        plt.savefig(f"./plots_{base}/dmd_{case}_v{v_ind + 1}_coeff.png")
        plt.tight_layout()
        plt.close()

        plt.figure()
        plt.plot(np.arange(1, N + 1), np.real(V_L[:, v_ind]), '-o')
        plt.axhline(y=0, linestyle=':', color='k')
        # plt.xticks(np.arange(1, N+1, 5), np.arange(1, N+1, 5))
        # plt.xticks([1, 100, 200, 300, 400], [1, 100, 200, 300, 400])
        plt.xlabel("Node number")
        plt.ylabel(f"Laplacian v{v_ind + 1}")
        plt.tight_layout()
        plt.savefig(f"./plots_{base}/Laplacian_{case}_v{v_ind + 1}.png")
        plt.close()

    # L changed
    T2_max = 50

    AdjM[2, 27] = 0
    AdjM[27, 2] = 0

    L2, d_M2, V_M2, d_L2, V_L2 = graph_laplacian_eigs(AdjM=AdjM, N=N, evs=evs)
    freq = np.angle(d_M2[:3])
    print(f"Actual Freq 2: {freq}")

    u2_static = generate_time_sequece(L=L2, N=N, T=51)
    dmdcoeff2_static, dominant_freqs2_static = graph_clustering_static(u=u2_static, d_M=d_M2, v_inds=v_inds, Nrows=Nrows, case=case, base=base, node_num=[2])

    u1 = append_time_sequence(u=u, L=L2, T=T2_max)
    # plot time series
    plt.figure()
    plt.plot(np.arange(T + T2_max + 1), u1[23, :], label="Node 24")
    plt.plot(np.arange(T + T2_max + 1), u1[24, :], label="Node 25")
    plt.legend()
    plt.savefig(f"./plots_{base}/u_{base}_T{T}_{T2_max}.png")
    plt.close()

    correct_array = np.zeros((T2_max, len(v_inds)))
    for T2 in range(T2_max):
        case2 = f"DMD_karate_T{T}_{T2 + 1}_rows{Nrows}"
        print(case2)
        u2 = u1[:, :T+T2+1]
        dmdcoeff2, dominant_freqs2 = graph_clustering_dynamic(u=u2, pre_dom_freqs=dominant_freqs, d_M=d_M2,
                                                              v_inds=v_inds,
                                                              Nrows=Nrows, case=case2, base=base, node_num=[2])

        df_output = pd.DataFrame(V_L2.real)
        for k_ind, v_ind in enumerate(v_inds):
            df_output[f"DMD_{v_ind}"] = dmdcoeff2[:, k_ind]
            dmd_correct = np.sum((df_output[v_ind].values * df_output[f"DMD_{v_ind}"].values > 0).astype(int))
            correct_array[T2, k_ind] = np.min([dmd_correct, N - dmd_correct]).astype(int)
            print(f"DMD vs Spectral v{v_ind + 1}: {correct_array[T2, k_ind]}")

        # df_output.to_csv(f"./output_{base}/clustering_{case2}.csv")

        for k_ind, v_ind in enumerate(v_inds):
            plt.figure()
            plt.plot(np.arange(1, N + 1), dmdcoeff2_static[:, k_ind], '-o')
            plt.plot(np.arange(1, N + 1), -dmdcoeff2[:, k_ind], '--s')
            plt.axhline(y=0, linestyle=':', color='k')
            # plt.xticks(np.arange(1, N+1, 5), np.arange(1, N+1, 5))
            # plt.xticks([1, 100, 200, 300, 400], [1, 100, 200, 300, 400])
            plt.xlabel("Node number")
            plt.ylabel("DMD Coefficient")
            plt.tight_layout()
            plt.savefig(f"./plots_{base}/dmd_{case2}_v{v_ind + 1}_coeff.png")
            plt.tight_layout()
            plt.close()

        # plt.figure()
        # plt.plot(np.arange(1, N + 1), np.real(V_L2[:, v_ind]), '-o')
        # plt.axhline(y=0, linestyle=':', color='k')
        # # plt.xticks(np.arange(1, N+1, 5), np.arange(1, N+1, 5))
        # # plt.xticks([1, 100, 200, 300, 400], [1, 100, 200, 300, 400])
        # plt.xlabel("Node number")
        # plt.ylabel(f"Laplacian v{v_ind + 1}")
        # plt.tight_layout()
        # plt.savefig(f"./plots/Laplacian_{case2}_v{v_ind + 1}.png")
        # plt.close()

    df_correct = pd.DataFrame(correct_array, columns=[f"mode {v_ind}" for v_ind in v_inds])
    df_correct.to_csv(f"./output_{base}/dynamic_clustering_DMD_karate_correctness.csv")

    plt.figure()
    plt.plot(np.real(d_M), np.imag(d_M), 's', alpha=0.8, label='original')
    plt.plot(np.real(d_M2), np.imag(d_M2), 'o', alpha=0.8, label='new')
    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.legend()
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"./plots_{base}/eigenvalues_karate_diff.png")
    plt.close()

    plt.figure()
    plt.plot(np.arange(1, N + 1), np.real(V_L[:, 1]), '-o')
    plt.plot(np.arange(1, N + 1), -np.real(V_L2[:, 1]), '--s')
    plt.axhline(y=0, linestyle=':', color='k')
    # plt.xticks(np.arange(1, N+1, 5), np.arange(1, N+1, 5))
    # plt.xticks([1, 100, 200, 300, 400], [1, 100, 200, 300, 400])
    plt.xlabel("Node number")
    plt.ylabel("Laplacian v2")
    plt.tight_layout()
    plt.savefig(f"./plots_{base}/Laplacian_karate_v2_diff.png")
    plt.close()
