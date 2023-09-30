from __future__ import annotations

import warnings
import time
import tqdm
from itertools import permutations
from queue import Queue
from typing import List, Set, Tuple, Dict
from numpy import ndarray

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Graph import Graph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node
from causallearn.utils.ChoiceGenerator import ChoiceGenerator
from causallearn.utils.DepthChoiceGenerator import DepthChoiceGenerator
from causallearn.utils.cit import *
from causallearn.utils.Fas import fas
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.search.ConstraintBased.FCI import *

from clustercausal.clusterdag.ClusterDAG import ClusterDAG


class ClusterFCI:
    def __init__(
        self,
        cdag: ClusterDAG,
        dataset: ndarray,
        independence_test_method: str = fisherz,
        alpha: float = 0.05,
        depth: int = -1,
        max_path_length: int = -1,
        verbose: bool = False,
        background_knowledge: BackgroundKnowledge | None = None,
        **kwargs,
    ):
        """
        Set parameters for the clust PC algorithm.
        """
        self.cdag = cdag
        self.dataset = dataset
        self.independence_test_method = independence_test_method
        self.alpha = alpha
        self.depth = depth
        self.max_path_length = max_path_length
        self.verbose = verbose
        self.background_knowledge = background_knowledge
        self.kwargs = kwargs
        self.cdag.cdag_to_mpdag()
        self.cdag.get_cluster_topological_ordering()  # Get topological ordering of CDAG

        # Set independence test
        self.cdag.cg.test = CIT(
            self.dataset, independence_test_method, **kwargs
        )

    def run(self):
        """
        Runs the C-FCI algorithm.
        """
        start = time.time()
        no_of_var = self.data.shape[1]
        assert len(self.cdag.node_names) == no_of_var
        if self.verbose:
            print(
                f"Topological ordering {(self.cdag.cdag_list_of_topological_sort)}"
            )
        sepsets = set()
        for cluster_name in self.cdag.cdag_list_of_topological_sort:
            graph, sepsets_clust = self.cluster_fas_phase(cluster_name)
            sepsets = sepsets.union(sepsets_clust)

        end = time.time()
        self.cdag.cg.PC_elapsed = end - start
        print(f"Duration of algorithm was {self.cdag.cg.PC_elapsed:.2f}sec")

    def cluster_fas_phase(self, cluster_name):
        """
        Runs the cluster phase of the C-FCI algorithm, which
        is an adapted fas adjacency search from causallearn
        """
        start_cluster = time.time()
        assert type(self.data) == np.ndarray
        assert 0 < self.alpha < 1
        depth = -1
        cluster = ClusterDAG.get_node_by_name(
            cluster_name, self.cdag.cluster_graph
        )
        cluster_node_indices = self.cdag.get_node_indices_of_cluster(cluster)
        cluster_node_indices = np.array(sorted(cluster_node_indices))
        local_graph = self.cdag.get_local_graph(cluster)

        # Only to check max degree
        local_graph_node_indices = np.array(
            [self.cdag.cg.G.node_map[node] for node in local_graph.G.nodes]
        )
        local_graph_node_indices = np.array(sorted(local_graph_node_indices))

        if self.verbose:
            print(
                f"Cluster node indices of {cluster.get_name()} are {cluster_node_indices}"
            )

        if self.verbose:
            print(
                f"Local graph node indices of {cluster.get_name()} are {local_graph_node_indices}"
            )
        if self.show_progress:
            if len(local_graph_node_indices) == 1:
                pbar = tqdm(total=1)
                pbar.reset()
                pbar.update()
                pbar.set_description(
                    f"{cluster.get_name()} phase, no nonchild, nothing to do"
                )
            else:
                pbar = tqdm(total=cluster_node_indices.shape[0])
