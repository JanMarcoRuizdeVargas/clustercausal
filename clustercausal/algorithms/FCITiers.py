from __future__ import annotations

import warnings
import time
from tqdm.auto import tqdm
from itertools import combinations, permutations
from queue import Queue
from typing import List, Set, Tuple, Dict
from numpy import ndarray

from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Graph import Graph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node

# from causallearn.search.ConstraintBased.FCI import SepsetsPossibleDsep
from causallearn.utils.PCUtils.Helper import append_value
from causallearn.utils.ChoiceGenerator import ChoiceGenerator
from causallearn.utils.DepthChoiceGenerator import DepthChoiceGenerator
from causallearn.utils.cit import *
from causallearn.utils.Fas import fas
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.search.ConstraintBased.FCI import *

from clustercausal.clusterdag.ClusterDAG import ClusterDAG


class FCITiers:
    def __init__(
        self,
        cdag: ClusterDAG,
        dataset: ndarray,
        independence_test_method: str = fisherz,
        stable: bool = True,
        alpha: float = 0.05,
        depth: int = -1,
        max_path_length: int = -1,
        verbose: bool = False,
        background_knowledge: BackgroundKnowledge | None = None,
        show_progress: bool = True,
        **kwargs,
    ):
        """
        Set parameters for the clust PC algorithm.
        """
        #
        self.cdag = cdag
        self.tiers = cdag.get_cluster_topological_ordering()
        self.cg = CausalGraph(
            no_of_var=len(cdag.node_names), node_names=cdag.node_names
        )

        # FCI Tiers starts from empty graph
        for edge in self.cg.G.get_graph_edges():
            self.cg.G.remove_edge(edge)

        self.dataset = dataset
        self.stable = stable
        self.alpha = alpha
        self.depth = depth
        self.max_path_length = max_path_length
        self.verbose = verbose
        self.background_knowledge = background_knowledge
        if background_knowledge is not None:
            raise ValueError("Background knowledge is not supported for now")
        self.show_progress = show_progress
        self.kwargs = kwargs

        # Set independence test for cluster_phase
        self.cdag.cg.test = CIT(
            self.dataset, independence_test_method, **kwargs
        )
        # Set independence test for run
        self.independence_test_method = CIT(
            self.dataset, independence_test_method, **kwargs
        )

    def run(self):
        """
        Run FCITiers algorithm
        """
        start = time.time()
        no_of_var = self.dataset.shape[1]
        self.no_of_indep_tests_performed = 0
        assert len(self.cdag.node_names) == no_of_var
        if self.verbose:
            print(
                f"Topological ordering {(self.cdag.cdag_list_of_topological_sort)}"
            )
        # self.cdag.cg.sepset = set()
        self.sep_sets: Dict[Tuple[int, int], Set[int]] = {}

        # As some nodes have no edge by CDAG definition, they never get tested so have Nonetype sepsets
        # manually have to add the parent set of i and parent set of j to sepset(i, j) and sepset(j, i)
        for i in range(no_of_var):
            for j in range(no_of_var):
                node_i = self.cdag.get_key_by_value(self.cdag.cg.G.node_map, i)
                node_j = self.cdag.get_key_by_value(self.cdag.cg.G.node_map, j)
                edge = self.cdag.cg.G.get_edge(node_i, node_j)
                if edge is None:
                    parents_i = self.cdag.cg.G.get_parents(node_i)
                    index_parents_i = [
                        self.cdag.cg.G.node_map[parent] for parent in parents_i
                    ]
                    parents_j = self.cdag.cg.G.get_parents(node_j)
                    index_parents_j = [
                        self.cdag.cg.G.node_map[parent] for parent in parents_j
                    ]
                    append_value(
                        self.cdag.cg.sepset, i, j, tuple(index_parents_i)
                    )
                    append_value(
                        self.cdag.cg.sepset, i, j, tuple(index_parents_j)
                    )
                    self.sep_sets[(i, j)] = set(index_parents_i).union(
                        index_parents_j
                    )
                    append_value(
                        self.cdag.cg.sepset, j, i, tuple(index_parents_i)
                    )
                    append_value(
                        self.cdag.cg.sepset, j, i, tuple(index_parents_j)
                    )
                    self.sep_sets[(j, i)] = set(index_parents_i).union(
                        index_parents_j
                    )
        n = len(self.tiers)
        for i in range(n, 0, -1):
            if self.verbose:
                print(f"Tier {i}")
            pass
            # Create A_i, B_i, O_i
            A_i = self.tiers[: i - 1]
            B_i = self.tiers[i - 1]
            # do FCI exogenous
            self.fci_exogenous(A_i, B_i)

        edges = get_color_edges(self.cdag.cg.G)

        return self.cg, edges

    def fci_exogenous(self, A_i, B_i):
        """
        FCI exogenous phase
        adds edges to self.cg
        """
        pass
        # TODO
