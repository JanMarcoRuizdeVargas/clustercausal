import numpy as np
import random
import causallearn
from clustercausal.clusterdag.ClusterDAG import ClusterDAG


def draw_graph(nodes, edges):
    artificial_mapping = {}
    for node_name in nodes:
        artificial_mapping[node_name] = [node_name]
    cdag = ClusterDAG(artificial_mapping, edges)
    cdag.cluster_graph.draw_pydot_graph()


def generate_gaussian_anm(
    nodes, edges, num_samples=10000, seed=None, edge_weights=None
):
    n = len(nodes)
    node_map = {}
    for i in range(n):
        node_map[nodes[i]] = i
    data = np.zeros((num_samples, n))
    rng = np.random.default_rng(seed)
    if edge_weights is None:
        edge_weights = {}
        for edge in edges:
            edge_weights[edge] = rng.choice([-3, -2, -1, 1, 2, 3])
    for node in nodes:
        influence = np.zeros(num_samples)
        for edge in edges:
            if edge[1] == node:
                influence += edge_weights[edge] * data[:, node_map[edge[0]]]
        sample = rng.normal(size=num_samples)
        data[:, node_map[node]] = (
            influence + sample
        )  # np.random.normal(size = num_samples)
    return data, edge_weights


def is_valid_clustering(cdag, causal_graph):
    """
    Checks if a CDAG is a valid clustering of a causal graph

    Parameters
    ----------
    cdag : CDAG
        instance of CDAG class
    causal_graph : CausalGraph
        instance of CausalGraph class
    """
    pass