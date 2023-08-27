from __future__ import annotations

import numpy as np
from numpy import ndarray
import pandas as pd
import networkx as nx
import causallearn

# import castle
import logging

import time
import warnings
from itertools import combinations, permutations
from typing import Dict, List, Tuple
from tqdm.auto import tqdm

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.Helper import append_value
from causallearn.utils.cit import *
from causallearn.utils.PCUtils import Helper, Meek, SkeletonDiscovery, UCSepset
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import (
    orient_by_background_knowledge,
)
from causallearn.search.ConstraintBased.PC import pc, pc_alg

from clustercausal.clusterdag.ClusterDAG import ClusterDAG

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class ClusterPC:
    """
    Runs the ClustPC algorithm according to my master thesis.
    Adapted from causallearn pc algorithm.

    """

    def __init__(
        self,
        cdag: ClusterDAG,
        data: ndarray,
        alpha: float,
        indep_test: str = "fisherz",
        stable: bool = True,
        uc_rule: int = 0,
        uc_priority: int = 2,
        background_knowledge: BackgroundKnowledge | None = None,
        verbose: bool = False,
        show_progress: bool = True,
        **kwargs,
    ) -> None:
        """
        Set parameters for the clust PC algorithm.
        """
        self.cdag = cdag
        self.data = data
        self.node_names = cdag.node_names
        self.alpha = alpha

        self.stable = stable
        self.uc_rule = uc_rule
        self.uc_priority = uc_priority
        self.background_knowledge = background_knowledge
        self.verbose = verbose
        self.show_progress = show_progress
        self.kwargs = kwargs
        self.cdag.cdag_to_mpdag()  # Updates self.cdag.cg to mpdag
        self.cdag.get_cluster_topological_ordering()  # saved in self.cdag.cdag_list_of_topological_sort

        self.cdag.cg.test = CIT(self.data, indep_test, **kwargs)

    def run(self) -> CausalGraph:
        start = time.time()
        no_of_var = self.data.shape[1]
        assert len(self.cdag.node_names) == no_of_var
        if self.verbose:
            print(
                f"Topological ordering {(self.cdag.cdag_list_of_topological_sort)}"
            )
        for cluster_name in self.cdag.cdag_list_of_topological_sort:
            self.cdag.cg = self.cluster_phase(cluster_name)

        # print("Applying edge orientation rules")
        # As some nodes have no edge by CDAG definition, they never get tested so have Nonetype sepsets
        # manually have to add an empty sepset for them else the Meek rules try to access NoneType
        for i in range(no_of_var):
            for j in range(no_of_var):
                append_value(self.cdag.cg.sepset, i, j, tuple(set()))
                append_value(self.cdag.cg.sepset, j, i, tuple(set()))
        cg_1 = self.cdag.cg
        background_knowledge = self.background_knowledge
        if self.uc_rule == 0:
            if self.uc_priority != -1:
                cg_2 = UCSepset.uc_sepset(
                    cg_1,
                    self.uc_priority,
                    background_knowledge=background_knowledge,
                )
            else:
                cg_2 = UCSepset.uc_sepset(
                    cg_1, background_knowledge=background_knowledge
                )
            cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

        elif self.uc_rule == 1:
            if self.uc_priority != -1:
                cg_2 = UCSepset.maxp(
                    cg_1,
                    self.uc_priority,
                    background_knowledge=background_knowledge,
                )
            else:
                cg_2 = UCSepset.maxp(
                    cg_1, background_knowledge=background_knowledge
                )
            cg = Meek.meek(cg_2, background_knowledge=background_knowledge)

        elif self.uc_rule == 2:
            if self.uc_priority != -1:
                cg_2 = UCSepset.definite_maxp(
                    cg_1,
                    self.alpha,
                    self.uc_priority,
                    background_knowledge=background_knowledge,
                )
            else:
                cg_2 = UCSepset.definite_maxp(
                    cg_1, self.alpha, background_knowledge=background_knowledge
                )
            cg_before = Meek.definite_meek(
                cg_2, background_knowledge=background_knowledge
            )
            cg = Meek.meek(
                cg_before, background_knowledge=background_knowledge
            )
        else:
            raise ValueError("uc_rule should be in [0, 1, 2]")
        self.cdag.cg.G.reconstitute_dpath(self.cdag.cg.G.get_graph_edges())
        self.cdag.cg = cg
        end = time.time()
        self.cdag.cg.PC_elapsed = end - start
        print(f"Duration of algorithm was {self.cdag.cg.PC_elapsed:.2f}sec")

        return self.cdag.cg  # Return CausalGraph of the CDAG

    def cluster_phase(self, cluster_name) -> CausalGraph:
        start_cluster = time.time()
        assert type(self.data) == np.ndarray
        assert 0 < self.alpha < 1

        depth = -1
        cluster = ClusterDAG.get_node_by_name(
            cluster_name, self.cdag.cluster_graph
        )
        cluster_node_indices = self.cdag.get_node_indices_of_cluster(cluster)
        cluster_node_indices = np.array(sorted(cluster_node_indices))
        local_graph = self.cdag.get_local_graph(
            cluster
        )  # Only to check max degree
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
        # Depth loop
        # Depends on max nonchilds within cluster and max degree of parents
        # of cluster, as sets in neighbors of top cluster nodes are considered
        # as possible separating sets
        while (
            self.cdag.max_nonchild_degree_of_cluster(cluster) - 1 > depth
        ) or (
            self.cdag.max_degree_of_cluster_parents_in_local_graph(
                cluster, local_graph
            )
            - 1
            > depth
        ):
            depth += 1
            depth_start = time.time()
            edge_removal = []
            if self.show_progress:
                pbar.reset()
            for x in cluster_node_indices:
                if self.show_progress:
                    pbar.update()
                if self.show_progress:
                    pbar.set_description(
                        f"{cluster.get_name()} phase, Depth={depth}, working on node {x}"
                    )
                Nonchilds_x = self.cdag.get_nonchilds(x)
                # Remove neighbors that are not in local_graph or are in cluster
                # cluster_mask = np.isin(Nonchilds_x, cluster_node_indices)
                # Pa_x_outside_cluster = Nonchilds_x[~cluster_mask]
                # local_mask = np.isin(Nonchilds_x, local_graph_node_indices)
                # Nonchilds_x_local_graph = Nonchilds_x[local_mask]
                if len(Nonchilds_x) < depth - 1:
                    continue
                if self.verbose:
                    print(f"Nonchilds of {x} are {Nonchilds_x}")
                for y in Nonchilds_x:
                    if self.verbose:
                        print("Testing edges from %d to %d" % (x, y))
                        # No other background knowledge functionality supported for now
                    sepsets = set()
                    Nonchilds_x_no_y = np.delete(
                        Nonchilds_x,
                        np.where(Nonchilds_x == y),
                    )

                    # Consider separating sets in nonchilds(x) intersect local_graph
                    for S in combinations(Nonchilds_x_no_y, depth):
                        # print(f'Set S to be tested is {S}')
                        p = self.cdag.cg.ci_test(x, y, S)
                        if p > self.alpha:
                            if self.verbose:
                                print(
                                    "%d ind %d | %s with p-value %f"
                                    % (x, y, S, p)
                                )
                            if not self.stable:
                                edge1 = self.cdag.cg.G.get_edge(
                                    self.cdag.cg.G.nodes[x],
                                    self.cdag.cg.G.nodes[y],
                                )
                                if edge1 is not None:
                                    self.cdag.cg.G.remove_edge(edge1)
                                edge2 = self.cdag.cg.G.get_edge(
                                    self.cdag.cg.G.nodes[y],
                                    self.cdag.cg.G.nodes[x],
                                )
                                if edge2 is not None:
                                    self.cdag.cg.G.remove_edge(edge2)
                                append_value(self.cdag.cg.sepset, x, y, S)
                                append_value(self.cdag.cg.sepset, y, x, S)
                                break
                            else:
                                edge_removal.append(
                                    (x, y)
                                )  # after all conditioning sets at
                                edge_removal.append(
                                    (y, x)
                                )  # depth l have been considered
                                for s in S:
                                    sepsets.add(s)
                        else:
                            if self.verbose:
                                print(
                                    "%d dep %d | %s with p-value %f"
                                    % (x, y, S, p)
                                )
                    # print(f'Added sepset: {x} !- {y} | {tuple(sepsets)}')
                    # print(f'Type of sepsets is {type(sepsets)}')
                    append_value(self.cdag.cg.sepset, x, y, tuple(sepsets))
                    append_value(self.cdag.cg.sepset, y, x, tuple(sepsets))
                    x_node = self.cdag.get_key_by_value(
                        self.cdag.cg.G.node_map, x
                    )
                    y_node = self.cdag.get_key_by_value(
                        self.cdag.cg.G.node_map, y
                    )
                    if y_node in self.cdag.cg.G.get_parents(x_node):
                        # Consider separating sets in neighbors(y)
                        Neigh_y = self.cdag.cg.neighbors(y)
                        # Remove neighbors that are not in local_graph or are in cluster
                        local_mask = np.isin(Neigh_y, local_graph_node_indices)
                        Neigh_y_in_local_graph = Neigh_y[local_mask]
                        if len(Neigh_y_in_local_graph) < depth - 1:
                            continue
                        if self.verbose:
                            print(
                                f"Neighbors of {y} in local graph are {Neigh_y_in_local_graph}"
                            )
                        Neighbors_y_no_x = np.delete(
                            Neigh_y_in_local_graph,
                            np.where(Neigh_y_in_local_graph == x),
                        )
                        for S in combinations(Neighbors_y_no_x, depth):
                            # print(f'Set S to be tested is {S}')
                            p = self.cdag.cg.ci_test(x, y, S)
                            if p > self.alpha:
                                if self.verbose:
                                    print(
                                        "%d ind %d | %s with p-value %f"
                                        % (x, y, S, p)
                                    )
                                if not self.stable:
                                    edge1 = self.cdag.cg.G.get_edge(
                                        self.cdag.cg.G.nodes[x],
                                        self.cdag.cg.G.nodes[y],
                                    )
                                    if edge1 is not None:
                                        self.cdag.cg.G.remove_edge(edge1)
                                    edge2 = self.cdag.cg.G.get_edge(
                                        self.cdag.cg.G.nodes[y],
                                        self.cdag.cg.G.nodes[x],
                                    )
                                    if edge2 is not None:
                                        self.cdag.cg.G.remove_edge(edge2)
                                    append_value(self.cdag.cg.sepset, x, y, S)
                                    append_value(self.cdag.cg.sepset, y, x, S)
                                    break
                                else:
                                    edge_removal.append(
                                        (x, y)
                                    )  # after all conditioning sets at
                                    edge_removal.append(
                                        (y, x)
                                    )  # depth l have been considered
                                    for s in S:
                                        sepsets.add(s)
                            else:
                                if self.verbose:
                                    print(
                                        "%d dep %d | %s with p-value %f"
                                        % (x, y, S, p)
                                    )
                        # print(f'Added sepset: {x} !- {y} | {tuple(sepsets)}')
                        # print(f'Type of sepsets is {type(sepsets)}')
                        append_value(self.cdag.cg.sepset, x, y, tuple(sepsets))
                        append_value(self.cdag.cg.sepset, y, x, tuple(sepsets))
            if self.show_progress:
                pbar.refresh()

            for x, y in list(set(edge_removal)):
                edge1 = self.cdag.cg.G.get_edge(
                    self.cdag.cg.G.nodes[x], self.cdag.cg.G.nodes[y]
                )
                if edge1 is not None:
                    self.cdag.cg.G.remove_edge(edge1)
                    x_name = self.cdag.get_key_by_value(
                        self.cdag.cg.G.node_map, x
                    )
                    y_name = self.cdag.get_key_by_value(
                        self.cdag.cg.G.node_map, y
                    )
                    if self.verbose:
                        print(f"Deleted edge from {x_name} to {y_name}")
            local_graph = self.cdag.get_local_graph(cluster)
            # print('LOCAL GRAPH DRAWN BELOW')
            # local_graph.draw_pydot_graph()
            depth_end = time.time()
            # print(f"Depth {depth} took {depth_end - depth_start:.2f}sec")
        end_cluster = time.time()
        time_elapsed = end_cluster - start_cluster
        if self.show_progress:
            pbar.set_postfix_str(
                f"duration: {time_elapsed:.2f}sec", refresh=True
            )
            pbar.close()

        return self.cdag.cg
