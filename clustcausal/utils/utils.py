import numpy as np
import causallearn
import random
from clustcausal.clusterdag.cluster_dag import CDAG

def draw_graph(nodes, edges):
    artificial_mapping = {}
    for node_name in nodes:
        artificial_mapping[node_name] = [node_name]
    cdag = CDAG(artificial_mapping, edges)
    cdag.cluster_graph.draw_pydot_graph()

def generate_gaussian_anm(nodes, edges, num_samples = 10000, edge_weights = None):
    n = len(nodes)
    node_map = {}
    for i in range(n):
        node_map[nodes[i]] = i
    data = np.zeros((num_samples, n))
    if edge_weights is None: 
        edge_weights = {}
        for edge in edges:
            edge_weights[edge] = random.choice([-3,-2,-1,1,2,3])
    for node in nodes:
        influence = np.zeros(num_samples)
        for edge in edges:
            if edge[1] == node:
                influence += edge_weights[edge]*data[:,node_map[edge[0]]]
        sample = np.random.normal(size = num_samples)
        data[:,node_map[node]] = influence + sample # np.random.normal(size = num_samples)
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