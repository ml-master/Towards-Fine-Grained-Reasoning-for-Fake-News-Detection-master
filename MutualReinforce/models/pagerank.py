"""PageRank for multiple graphs. """
import networkx as nx
import numpy as np
import time
np.set_printoptions(precision=3)

__all__ = ["get_pagerank_score", "pagerank", "pagerank_numpy", "google_matrix"]

def get_pagerank_score(G_triple, filename, draw_gephi=False):
    pr = pagerank_numpy(G_triple)
    t0 = time.process_time()
    pr_pers = pagerank_numpy(G_triple, do_personalization=True)
    t1 = time.process_time()
    return pr, pr_pers, t1-t0

    if draw_gephi:
        nx.write_gexf(G_triple, f'{filename}.gexf')


def pagerank(
        G,
        alpha=0.85,
        personalization=None,
        max_iter=100,
        tol=1.0e-6,
        nstart=None,
        weight="weight",
        dangling=None,
        partition=None,
        params=None
):
    if len(G) == 0:
        return {}

    # params = {
    #     "a_tt": 0.85,
    #     "a_ut": 0.85,
    #     "a_tu": 0.85,
    #     "a_uu": 0.85,
    # }
    if not params:

        a_tt = 1
        a_tk = 1
        a_kt = 1
        a_kk = 1

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = {k: v / s for k, v in nstart.items()}

    if personalization is None:
        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        s = float(sum(personalization.values()))
        p = {k: v / s for k, v in personalization.items()}

    if dangling is None:
        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        s = float(sum(dangling.values()))
        dangling_weights = {k: v / s for k, v in dangling.items()}
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    nodes_twt = [x for x, y in G.nodes(data=True) if y['type'] == 'twt']
    nodes_key = [x for x, y in G.nodes(data=True) if y['type'] == 'key']

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)

        #
        for n in nodes_twt:
            # tweet -> tweet
            for nbr in sorted(list(set(W[n]) & set(nodes_twt))):
                x[nbr] += a_tt * alpha * xlast[n] * W[n][nbr][weight]

            # tweet -> key
            for nbr in set(W[n]) & set(nodes_key):
                x[nbr] += a_tk * alpha * xlast[n] * W[n][nbr][weight]

            x[n] += danglesum * dangling_weights.get(n, 0) + (1.0 - alpha) * p.get(n, 0)

        for n in nodes_key:
            # key -> tweet
            for nbr in sorted(list(set(W[n]) & set(nodes_twt))):
                x[nbr] += a_kt * alpha * xlast[n] * W[n][nbr][weight]

            # key -> key
            for nbr in set(W[n]) & set(nodes_key):
                x[nbr] += a_kk * alpha * xlast[n] * W[n][nbr][weight]

            x[n] += danglesum * dangling_weights.get(n, 0) + (1.0 - alpha) * p.get(n, 0)

        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x
    raise nx.PowerIterationFailedConvergence(max_iter)


def google_matrix(
        G, alpha=0.85, personalization=None, weight="weight", dangling=None, partition=None, params=None, do_personalization=False
):
    import numpy as np

    a_tt, a_tk, a_tu = 1,    0.5, 0.25
    a_kt, a_kk, a_ku = 0.5,    1,  0.5
    a_ut, a_uk, a_uu = 0.25, 0.5,    1

    if params is not None:
        a_tt, a_tk, a_tu = params[0], params[1], params[2]
        a_kt, a_kk, a_ku = params[3], params[4], params[5]
        a_ut, a_uk, a_uu = params[6], params[7], params[8]


    if do_personalization:
        personalization = {node_id: d['score'] for node_id, d in G.nodes(data=True)}


    nodes_twt = [x for x, y in G.nodes(data=True) if y['type'] == 'twt']
    nodes_key = [x for x, y in G.nodes(data=True) if y['type'] == 'key']
    nodes_usr = [x for x, y in G.nodes(data=True) if y['type'] == 'usr']

    nodelist = nodes_twt + nodes_key + nodes_usr
    assert len(nodelist) == len(G)

    M = nx.to_numpy_matrix(G, nodelist=nodelist, weight=weight)
    n_twt = len(nodes_twt)
    n_key = len(nodes_key)

    M[     :n_twt        ,      :n_twt        ] *= a_tt
    M[     :n_twt        , n_twt:n_twt + n_key] *= a_tk
    M[n_twt:n_twt + n_key,      :n_twt        ] *= a_kt
    M[n_twt:n_twt + n_key, n_twt:n_twt + n_key] *= a_kk

    if nodes_usr != []:
        M[     :n_twt        ,n_twt + n_key:     ] *= a_tu
        M[n_twt:n_twt + n_key,n_twt + n_key:     ] *= a_ku

        M[n_twt + n_key:     ,     :n_twt        ] *= a_ut
        M[n_twt + n_key:     ,n_twt:n_twt + n_key] *= a_uk
        M[n_twt + n_key:     ,n_twt + n_key:     ] *= a_uu

    N = len(G)
    if N == 0:
        return M

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        p /= p.sum()

    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    dangling_nodes = np.where(M.sum(axis=1) == 0)[0]

    # Assign dangling_weights to any dangling nodes (nodes with no out links)
    for node in dangling_nodes:
        M[node] = dangling_weights

    # Normalize rows to sum to 1
    M /= M.sum(axis=1)

    return alpha * M + (1 - alpha) * p


def pagerank_numpy(G, alpha=0.85, personalization=None, weight="weight", dangling=None, do_personalization=False):
    import numpy as np

    if len(G) == 0:
        return {}
    M = google_matrix(
        G, alpha, personalization=personalization, weight=weight, dangling=dangling, do_personalization=do_personalization
    )

    eigenvalues, eigenvectors = np.linalg.eig(M.T)
    ind = np.argmax(eigenvalues)
    # eigenvector of largest eigenvalue is at ind, normalized
    largest = np.array(eigenvectors[:, ind]).flatten().real
    norm = float(largest.sum())
    return dict(zip(G, map(float, largest / norm)))
