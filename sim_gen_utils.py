""" This file contains utility functions for the simulation generator."""
import networkx as nx
import numpy as np
import datetime
import os
import random
import torch


def get_time_stamped_outdir(path,
                            results_file_postfix,
                            time_format='%Y%m%dT%H%M%S.%f'):
    ''' Creates a directory with the current time stamp and the given path.'''
    time_str = datetime.datetime.now().strftime(time_format)
    if len(results_file_postfix) > 0:
        time_str = time_str + '_' + results_file_postfix
    path.append(time_str)
    res = get_dir_from_path_list(path)
    return res


def get_dir_from_path_list(path):
    ''' Creates a directory with the given path.'''
    outdir = path[0]
    if not (os.path.exists(outdir)):
        os.makedirs(outdir)
    for p in path[1:]:
        outdir = os.path.join(outdir, p)
        if not (os.path.exists(outdir)):
            os.makedirs(outdir)
    return outdir


def custom_watts_strogatz_graph(n, k, p, seed=random):
    """Returns a Watts–Strogatz small-world graph.
    Parameters
    ----------
    n : int
        The number of nodes
    k : int
        Each node is joined with its `k` nearest neighbors in a ring
        topology.
    p : float
        The probability of rewiring each edge
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
        Have set the seed as 'random'. The actual seed is set in the main.py code
    See Also
    --------
    newman_watts_strogatz_graph()
    connected_watts_strogatz_graph()
    Notes
    -----
    First create a ring over $n$ nodes [1]_.  Then each node in the ring is joined
    to its $k$ nearest neighbors (or $k - 1$ neighbors if $k$ is odd).
    Then shortcuts are created by replacing some edges as follows: for each
    edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest neighbors"
    with probability $p$ replace it with a new edge $(u, w)$ with uniformly
    random choice of existing node $w$.
    In contrast with :func:`newman_watts_strogatz_graph`, the random rewiring
    does not increase the number of edges. The rewired graph is not guaranteed
    to be connected as in :func:`connected_watts_strogatz_graph`.
    References
    ----------
    .. [1] Duncan J. Watts and Steven H. Strogatz,
       Collective dynamics of small-world networks,
       Nature, 393, pp. 440--442, 1998.
    """
    p, node_names = p
    # node_names = list(range(50, 75))

    if k >= n:
        print("K: ", k, " N: ", n)
        raise nx.NetworkXError("k>=n, choose smaller k or larger n")

    G = nx.Graph()
    nodes = node_names  # list(range(n))  # nodes are labeled 0 to n-1
    # connect each node to k/2 neighbors
    for j in range(1, k // 2 + 1):
        assert p != 0 and k != 0
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        G.add_edges_from(zip(nodes, targets))
    # rewire edges from each node
    # loop over all nodes in order (label) and neighbors in order (distance)
    # no self loops or multiple edges allowed
    for j in range(1, k // 2 + 1):  # outer loop is neighbors
        assert p != 0 and k != 0
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        # inner loop in node order
        for u, v in zip(nodes, targets):
            if seed.random() < p:
                w = seed.choice(nodes)
                # Enforce no self-loops or multiple edges
                while w == u or G.has_edge(u, w):
                    w = seed.choice(nodes)
                    if G.degree(u) >= n - 1:
                        break  # skip this rewiring
                else:
                    G.remove_edge(u, v)
                    G.add_edge(u, w)
    return G

def normal_watts_strogatz_graph(n, agents, mu, sigma, seed=random):
    if mu >= n:
        print("mu: ", mu, " N: ", n)
        raise nx.NetworkXError("mu>=n, choose smaller mu or larger n")
    
    G = nx.Graph()
    nodes = agents  # list(range(n))  # nodes are labeled 0 to n-1

    # Generate degrees for each node based on normal distribution
    degrees = np.random.normal(loc=mu, scale=sigma, size=n).astype(int)

    # Ensure valid degrees (at least 1 and less than N)
    degrees = np.clip(degrees, 1, n - 1)
    
    # Connect nodes based on their assigned degree
    for i, node in enumerate(nodes):
        degree = degrees[i]  # Get the number of neighbors for this node
        targets = nodes[i+1:i+1+degree] + nodes[:max(0, (i+1+degree) - len(nodes))]  # Circular shift for neighbors
        G.add_edges_from(zip([node] * len(targets), targets))
    
    return G

# def make_one_hot(tensor, num_dim):
#     ''' Converts a tensor of indices to a one-hot tensor.'''
#     onehot = torch.zeros(tensor.shape[0], num_dim)
#     onehot.scatter_(1, tensor.view(-1, 1).long(), 1)
#     return onehot


def time_dists_split(i, f):
    ''' Returns a function that splits a string by comma and applies the function f to the i-th element.'''
    return lambda x: f(x[1:-1].strip().split(',')[i])