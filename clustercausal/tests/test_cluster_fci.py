import numpy as np
import causallearn
import networkx as nx

from causallearn.search.ConstraintBased.FCI import fci
from clustercausal.clusterdag.ClusterDAG import ClusterDAG
from clustercausal.algorithms.ClusterFCI import ClusterFCI
from clustercausal.experiments.Simulator import Simulator
from clustercausal.experiments.Evaluator import Evaluator
from clustercausal.utils.Utils import *


def test_clust_fci_to_base_fci():
    """
    Test case to check correct functioning of ClusterFCI
    """
    # List of node names
    node_names = ["X1", "X2", "X3", "U1", "U2"]

    # Parent dictionary
    parent_dict = {
        "X1": ["U1"],  # X1 has no parents
        "X2": ["U1", "U2"],  # X2 has X1 as a parent
        "X3": ["U2"],
        "U1": [],
        "U2": [],
    }

    # Call the make_graph function from Utils
    truth, W = make_graph(node_names, parent_dict)
    X = gaussian_data(W, 10000, seed=42)
    # remove confounders from X
    X = X[:, :3]
    cluster_dag = ClusterDAG(
        cluster_mapping={"C1": ["X1", "X2", "X3"]}, cluster_edges=[]
    )
    cluster_fci = ClusterFCI(cluster_dag, X, alpha=0.05, verbose=False)
    cluster_cg, cluster_edges = cluster_fci.run()

    base_G, base_edges = fci(X, alpha=0.05, verbose=False)
    base_cg = CausalGraph(3)
    base_cg.G = base_G
    assert cluster_cg.G.get_graph_edges() == base_cg.G.get_graph_edges()
