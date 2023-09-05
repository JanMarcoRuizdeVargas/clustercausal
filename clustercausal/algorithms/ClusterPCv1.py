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
        self.cdag.cdag_to_mpdag()  # Get pruned MPDAG from CDAG
        self.cdag.get_cluster_topological_ordering()  # Get topological ordering of CDAG

        # Set independence test
        self.cdag.cg.test = CIT(self.data, indep_test, **kwargs)

    def run(self) -> CausalGraph:
        """
        Runs the C-PC algorithm.
        Updates self.cdag.cg each step, which is a CausalGraph object.
        """
        start = time.time()
        no_of_var = self.data.shape[1]
        # pbar = tqdm(total=no_of_var) if self.show_progress else None
        if self.verbose:
            print(
                f"Topological ordering {(self.cdag.cdag_list_of_topological_sort)}"
            )
        for cluster_name in self.cdag.cdag_list_of_topological_sort:
            # print(f"\nBeginning work on cluster {cluster_name}")
            cluster = self.cdag.get_node_by_name(
                cluster_name, cg=self.cdag.cluster_graph
            )
            # for parent in self.cdag.cluster_graph.G.get_parents(cluster):
            #     # print(
            #     #     "\nInter phase between low cluster"
            #     #     f" {cluster.get_name()} and parent {parent.get_name()}"
            #     # )
            #     if parent is not None:
            #         self.inter_cluster_phase(cluster, parent)
            self.into_cluster_phase(cluster)
            # TODO Apply Meek edge orientation rules here too?
            # print(f"\nIntra phase in cluster {cluster.get_name()}")
            self.in_cluster_phase(cluster)
            # TODO Apply Meek edge orientation rules here too?
        # if self.show_progress:
        #   pbar.close()

        # TODO Meek edge orientation rules
        # Add d-separations present in cluster_graph to sepsets - cluster triplets
        # This code is not necessary - ignore
        """
        # This loop could be done more efficiently by only going through parents TODO
        c_edges = self.cdag.cluster_edges
        #c1, c2, cm are the names of clusters
        if self.show_progress:
            no_of_combinations = len(self.cdag.cluster_mapping.keys())**2 - len(self.cdag.cluster_mapping.keys())
            pbar = tqdm(range(no_of_combinations))
        for c1 in list(self.cdag.cluster_mapping.keys()):
            c1_cluster = self.cdag.get_node_by_name(c1, cg = self.cdag.cluster_graph)
            c1_indice = self.cdag.cluster_graph.G.node_map[c1_cluster]
            for c2 in list(self.cdag.cluster_mapping.keys()):
                c2_cluster = self.cdag.get_node_by_name(c2, cg = self.cdag.cluster_graph)
                c2_indice = self.cdag.cluster_graph.G.node_map[c2_cluster]
                if self.show_progress:
                    pbar.update()
                if self.show_progress:
                    pbar.set_description(f'Searching sepset between {c1} and {c2}')
                # If c1 is not c2 and they are not adjacent, they are d-separable
                # Find this set and add it to sepset for each node pair of c1, c2
                # (needed for Meeks rules, which need access to this information)
                if (c1 != c2) and (c1_indice not in self.cdag.cluster_graph.neighbors(c2_indice)):
                    # Find the cluster indices in cluster_graph of the potential separating clusters
                    depth = -1
                    delete = [c1_indice, c2_indice]
                    all_cluster_indices = np.array(range(self.cdag.cluster_graph.G.num_vars))
                    mask = np.isin(all_cluster_indices, delete)
                    other_cluster_indices = all_cluster_indices[~mask]
                    Flag = True
                    while self.cdag.cluster_graph.max_degree() - 1 > depth: # Search sepsets of cardinality 1,2,3,...
                      depth += 1
                      while Flag: # While not having found a separating set
                        for candidate_sepset in list(combinations(other_cluster_indices, depth)):
                          # candidate_sepset is list of cluster_indices, transform it into cluster_list (Node objects)
                          candidate_cluster_list = []
                          for indice in candidate_sepset:
                            dictionary = self.cdag.cluster_graph.G.node_map
                            candidate_cluster_list.append(self.cdag.get_key_by_value(dictionary, indice))
                          if self.cdag.cluster_graph.G.is_dseparated_from(c1_cluster, c2_cluster, candidate_cluster_list):
                            if self.verbose: print(f'{c1} d-connected {c2} | {candidate_sepset}')
                            # If d-separation is found in cluster_graph, get node_indices and add them to sepset of each i,j
                            # of the separated clusters
                            c1_node_indices = self.cdag.get_node_indices_of_cluster(c1_cluster)
                            c2_node_indices = self.cdag.get_node_indices_of_cluster(c2_cluster) 
                            potential_sepset = []
                            for candidate in candidate_sepset:
                              sepset_node_indices = self.cdag.get_node_indices_of_cluster(candidate)
                              potential_sepset.extend(sepset_node_indices)
                            for i in c1_node_indices:
                                for j in c2_node_indices:
                                    append_value(cg_0.sepset, i, j, sepset_node_indices)
                            Flag = False                           
        """

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

    # @profile
    def into_cluster_phase(self, low_cluster):
        """
        Runs the into-cluster phase of the ClustPC algorithm for a given cluster.
        All edges between low_cluster and pa(low_cluster) are considered at the
        same time.
        Input:
            -low_cluster: a node object of CausalGraph
        Updates self.cdag.cg each step.
        """
        start_into = time.time()
        assert type(self.data) == np.ndarray
        assert 0 < self.alpha < 1

        no_of_var = self.data.shape[1]
        # Check if all variables are in the graph
        assert len(self.cdag.node_names) == no_of_var

        # self.cdag.cg.set_ind_test(self.indep_test)

        depth = -1

        # Define the subgraph induced by the cluster nodes
        # Could be replaced by cdag.get_node_indices_of_cluster(cluster)
        nodes_names_in_low_cluster = self.cdag.cluster_mapping[
            low_cluster.get_name()
        ]
        nodes_names_in_high_clusters = []
        for parent in self.cdag.cluster_graph.G.get_parents(low_cluster):
            names = self.cdag.cluster_mapping[parent.get_name()]
            nodes_names_in_high_clusters.extend(names)
        # nodes_names_in_high_cluster = self.cdag.cluster_mapping[
        #     high_cluster.get_name()
        # ]
        nodes_in_low_cluster = self.cdag.get_list_of_nodes_by_name(
            list_of_node_names=nodes_names_in_low_cluster, cg=self.cdag.cg
        )
        nodes_in_high_clusters = self.cdag.get_list_of_nodes_by_name(
            list_of_node_names=nodes_names_in_high_clusters, cg=self.cdag.cg
        )
        # nodes_in_high_cluster = self.cdag.get_list_of_nodes_by_name(
        #     list_of_node_names=nodes_names_in_high_cluster, cg=self.cdag.cg
        # )
        # subgraph_cluster = CausalGraph(
        #     len(nodes_names_in_low_cluster) + len(nodes_names_in_high_cluster),
        #     nodes_names_in_low_cluster + nodes_names_in_high_cluster,
        # )
        # subgraph_cluster.G = self.cdag.subgraph(
        #     nodes_in_low_cluster + nodes_in_high_cluster
        # )

        # Collect the indices of nodes in subgraph and local_graph, w.r.t.
        # the entire adjacency matrix self.cdag.cg.G.graph
        cluster_node_indices = np.array(
            [self.cdag.cg.G.node_map[node] for node in nodes_in_low_cluster]
        )
        # Define the local graph which contains possible separating sets, here it is
        # low cluster union high cluster union low cluster parents
        # (high cluster is in low cluster parents per definition)
        local_graph = self.cdag.get_local_graph(low_cluster)
        local_graph_node_indices = np.array(
            [self.cdag.cg.G.node_map[node] for node in local_graph.G.nodes]
        )
        local_graph_node_indices = np.array(sorted(local_graph_node_indices))
        cluster_node_indices = np.array(sorted(cluster_node_indices))
        if self.verbose:
            print(
                f"Cluster node indices of {low_cluster.get_name()} are {cluster_node_indices}"
            )

        if self.verbose:
            print(
                f"Local graph node indices of {low_cluster.get_name()} are {local_graph_node_indices}"
            )
        if self.show_progress:
            if len(nodes_in_high_clusters) > 0:
                pbar = tqdm(total=cluster_node_indices.shape[0])
            else:
                pbar = tqdm(total=1)

        if (
            self.show_progress
        ):  # in case of only one node in cluster and no parent, close manually
            # if len(cluster_node_indices) == 1:
            #     x = cluster_node_indices[0]
            #     pbar.reset()
            #     pbar.update()
            #     pbar.set_description(
            #         f"Into: ->{low_cluster.get_name()}, Depth={depth}, working on node {x}"
            #     )
            # For highest clusters, there are no parents, so we have to close manually
            if len(nodes_in_high_clusters) == 0:
                pbar.reset()
                pbar.update()
                pbar.set_description(
                    f"Into: ->{low_cluster.get_name()}, no parents, nothing to do  "
                )
        # pbar = pbar

        # Difference to local pc algorithm is that we consider only edges in the cluster
        # but as potential separating sets we consider cluster union cluster parents
        # Therefore stopping criterion is when separating set cardinality
        # exceed nonchilds in cluster
        # Skeleton discovery
        end_prep = time.time()
        # print(f"Preparation time was {end_prep - start_into:.2f}sec")
        while (
            self.cdag.max_nonchilds_of_cluster_nodes(low_cluster, local_graph)
            - 1
            > depth
            and len(nodes_in_high_clusters) > 0
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
                        f"Into: ->{low_cluster.get_name()}, Depth={depth}, working on node {x}"
                    )
                # Get all nonchilds of node_x in the entire cg,  which is same as
                # neighbors in cluster union parents of cluster
                # format is integer values in adjacency matrix
                # Neigh_x = self.cdag.cg.neighbors(x)
                # node_x = ClusterDAG.get_key_by_value(
                #     self.cdag.cg.G.node_map, x
                # )
                # Child_x_nodes = self.cdag.cg.G.get_children(node_x)
                # Child_x_indices = [
                #     self.cdag.cg.G.node_map[node] for node in Child_x_nodes
                # ]
                # Nonchilds_x = np.setdiff1d(Neigh_x, Child_x_indices)
                Nonchilds_x = self.cdag.get_nonchilds(x)
                # if self.verbose: print(f'Neighbors of {x} is {Neigh_x}')
                # Remove neighbors that are not in local_graph or are in cluster
                cluster_mask = np.isin(Nonchilds_x, cluster_node_indices)
                Pa_x_outside_cluster = Nonchilds_x[~cluster_mask]
                local_mask = np.isin(Nonchilds_x, local_graph_node_indices)
                Nonchilds_x_local_graph = Nonchilds_x[local_mask]
                # Neigh_x_in_clust = Neigh_x[cluster_mask]
                if len(Nonchilds_x) < depth - 1:
                    continue
                if self.verbose:
                    print(
                        f"Parents of {x} in pa({low_cluster.get_name()})"
                        f" are {Pa_x_outside_cluster}"
                    )
                # local_mask = np.isin(Neigh_x, local_graph_node_indices)
                # possible_blocking_nodes = Neigh_x[local_mask]

                for y in Pa_x_outside_cluster:
                    if self.verbose:
                        print("Testing edges from %d to %d" % (x, y))
                    # No other background knowledge functionality supported for now
                    # No parent checking for mns separation supported for now TODO
                    sepsets = set()
                    Nonchilds_x_local_graph_no_y = np.delete(
                        Nonchilds_x_local_graph,
                        np.where(Nonchilds_x_local_graph == y),
                    )
                    for S in combinations(Nonchilds_x_local_graph_no_y, depth):
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
            local_graph = self.cdag.get_local_graph(low_cluster)
            # print('LOCAL GRAPH DRAWN BELOW')
            # local_graph.draw_pydot_graph()
            depth_end = time.time()
            # print(f"Depth {depth} took {depth_end - depth_start:.2f}sec")
        end_into = time.time()
        time_elapsed = end_into - start_into
        if self.show_progress:
            pbar.set_postfix_str(
                f"duration: {time_elapsed:.2f}sec", refresh=True
            )
            pbar.close()

        # TODO Orientation rules
        return self.cdag.cg

    def in_cluster_phase(self, cluster):
        """
        Runs the in-cluster phase of the ClustPC algorithm for a given cluster.
        Adapted from causallearn pc algorithm.
        Updates self.cdag.cg each step.
        Input: cluster, a node object of CausalGraph
        """
        start_in = time.time()
        assert type(self.data) == np.ndarray
        assert 0 < self.alpha < 1

        no_of_var = self.data.shape[1]
        # Check if all variables are in the graph
        assert len(self.cdag.node_names) == no_of_var

        # self.cdag.cg.set_ind_test(self.indep_test)

        depth = -1

        # Collect relevant nodes, i.e. cluster and parents of cluster in a list of Node objects

        # Define the subgraph induced by the cluster nodes
        # Could be replaced by cdag.get_node_indices_of_cluster(cluster)
        nodes_names_in_cluster = self.cdag.cluster_mapping[cluster.get_name()]
        nodes_in_cluster = self.cdag.get_list_of_nodes_by_name(
            list_of_node_names=nodes_names_in_cluster, cg=self.cdag.cg
        )
        subgraph_cluster = CausalGraph(
            len(nodes_names_in_cluster), nodes_names_in_cluster
        )
        subgraph_cluster.G = self.cdag.subgraph(nodes_in_cluster)
        cluster_node_indices = np.array(
            [
                self.cdag.cg.G.node_map[node]
                for node in subgraph_cluster.G.nodes
            ]
        )

        # Define the local graph which contains possible separating sets
        local_graph = self.cdag.get_local_graph(cluster)
        # Possibly replace subgraph_cluster and local_graph by index_arrays,
        # as have to operate on entire adjacency matrix and data matrix
        if self.verbose:
            print(
                f"Cluster node indices of {cluster.get_name()} are {cluster_node_indices}"
            )
        local_graph_node_indices = np.array(
            [self.cdag.cg.G.node_map[node] for node in local_graph.G.nodes]
        )
        if self.verbose:
            print(
                f"Local graph node indices of {cluster.get_name()} are {cluster_node_indices}"
            )

        pbar = (
            tqdm(total=cluster_node_indices.shape[0])
            if self.show_progress
            else None
        )

        if (
            self.show_progress
        ):  # in case of only one node in cluster, close manually
            if len(cluster_node_indices) == 1:
                x = cluster_node_indices[0]
                pbar.reset()
                pbar.update()
                pbar.set_description(
                    f"In:     {cluster.get_name()}, Depth={0}, working on node {x}"
                )
        # pbar = pbar

        # Difference to local pc algorithm is that we consider only edges in the cluster
        # but as potential separating sets we consider cluster union cluster parents
        # Therefore stopping criterion is when separating set cardinality
        # exceed nonchilds in cluster

        # Skeleton discovery
        while (
            self.cdag.max_nonchilds_of_cluster_nodes(cluster, local_graph) - 1
            > depth
        ):
            depth += 1
            edge_removal = []
            if self.show_progress:
                pbar.reset()
            for x in cluster_node_indices:
                if self.show_progress:
                    pbar.update()
                if self.show_progress:
                    pbar.set_description(
                        f"In:     {cluster.get_name()}, Depth={0}, working on node {x}"
                    )
                # Get all neighbors of node_x in the cluster, is integer values in adjacency matrix
                Neigh_x = self.cdag.cg.neighbors(x)
                # if self.verbose: print(f'Neigh_x is {Neigh_x}')
                # Remove neighbors that are not in cluster
                cluster_mask = np.isin(Neigh_x, cluster_node_indices)
                Neigh_x_in_clust = Neigh_x[cluster_mask]
                if self.verbose:
                    print(
                        f"Neighbors of {x} in {cluster.get_name()} are"
                        f" {Neigh_x_in_clust}"
                    )
                # Possible blocking nodes are
                local_mask = np.isin(Neigh_x, local_graph_node_indices)
                possible_blocking_nodes = Neigh_x[local_mask]
                # if self.verbose:
                #     print(
                #         f"Possible blocking nodes are {possible_blocking_nodes}"
                #     )
                if len(Neigh_x) < depth - 1:
                    continue
                for y in Neigh_x_in_clust:
                    if self.verbose:
                        print("Testing edges from %d to %d" % (x, y))
                    # No other background functionality supported for now
                    # No parent checking for mns separation supported for now TODO
                    sepsets = set()
                    Neigh_x_no_y = np.delete(
                        possible_blocking_nodes,
                        np.where(possible_blocking_nodes == y),
                    )
                    for S in combinations(Neigh_x_no_y, depth):
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
                    if self.verbose:
                        print(f"Deleted edge from {x} to {y}")
            # Update local graph to reflect edge deletions that were just done
            local_graph = self.cdag.get_local_graph(cluster)
            # print('LOCAL GRAPH DRAWN BELOW')
            # local_graph.draw_pydot_graph()
        end_in = time.time()
        time_elapsed = end_in - start_in
        if self.show_progress:
            pbar.set_postfix_str(
                f"duration: {time_elapsed:.2f}sec", refresh=True
            )
            pbar.close()

        # TODO Orientation rules
        return self.cdag.cg
