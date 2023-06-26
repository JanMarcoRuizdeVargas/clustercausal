from __future__ import annotations

import numpy as np
from numpy import ndarray
import pandas as pd
import networkx as nx
import causallearn
import castle
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
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
    orient_by_background_knowledge
from causallearn.search.ConstraintBased.PC import pc, pc_alg

from clustcausal.clusterdag.cluster_dag import CDAG

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ClustPC():
    """
    Runs the ClustPC algorithm according to my master thesis.
    Adapted from causallearn pc algorithm. 
    
    """
    # def __init__(self, cdag : CDAG):#, data: ndarray):
    #     self.cdag = cdag
    #     self.cdag.cdag_to_mpdag() # Get pruned MPDAG from CDAG
    #     self.cdag.get_topological_ordering()  # Get topological ordering of CDAG
    #     # self.data = data
    #     # if data is not pd.DataFrame:
    #     #     logging.info('Data converted into pandas dataframe with columns X1, X2, ...')
    #     #     data = pd.DataFrame(data)
    #     #     data.columns = self.cdag.node_names

    def __init__(self,
        cdag: CDAG,
        data: ndarray,
        # node_names: List[str] | None,
        alpha: float,
        indep_test: str = 'fisherz',
        stable: bool = True,
        uc_rule: int = 0,
        uc_priority: int = 2,
        background_knowledge: BackgroundKnowledge | None = None,
        verbose: bool = False,
        show_progress: bool = True,
        **kwargs
        ) -> None:
        """
        Set parameters for the clust PC algorithm.
        """
        self.cdag  = cdag
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
        self.cdag.cdag_to_mpdag() # Get pruned MPDAG from CDAG
        self.cdag.get_cluster_topological_ordering()  # Get topological ordering of CDAG

        # Set independence test
        self.cdag.cg.test = CIT(self.data, indep_test, **kwargs)
        # self.indep_test = indep_test
        # self.cdag.cg.set_ind_test(cit)
        # print('indep_test: ', type(self.cdag.cg.test), self.cdag.cg.test)
                       

    def run(self) -> CausalGraph:
        for cluster_name in self.cdag.cdag_list_of_topological_sort:
            cluster = self.cdag.get_node_by_name(cluster_name, cg = self.cdag.cluster_graph)
            for parent in self.cdag.cluster_graph.G.get_parents(cluster):
              self.inter_cluster_phase(cluster, parent)
            # Apply Meek edge orientation rules
            self.intra_cluster_phase(cluster)
            # Apply Meek edge orientation rules
        # Apply Meek edge orientation rules
        return self.cdag.cg # Return CausalGraph of the CDAG

    def inter_cluster_phase(self, low_cluster, high_cluster):
        """
        Runs the inter-cluster phase of the ClustPC algorithm for a given cluster.
        cluster is a node object of CausalGraph
        """
        # Identify the nodes needed to be considered, i.e. cluster and parents of cluster
        # parent_clusters = self.cdag.cluster_dag.G.get_parents(cluster)
        # self.cdag.cg.G.get_parents

        assert type(self.data) == np.ndarray
        assert 0 < self.alpha < 1
        
        no_of_var = self.data.shape[1]
        # Check if all variables are in the graph
        assert len(self.cdag.node_names) == no_of_var 

        # self.cdag.cg.set_ind_test(self.indep_test)

        depth = -1
        pbar = tqdm(total=no_of_var) if self.show_progress else None

        # Collect relevant nodes, i.e. clusters and parents of lower 
        # cluster in a list of Node objects
        # relevant_clusters, relevant_nodes = self.cdag.get_parent_plus(cluster)

        # Define the subgraph induced by the cluster nodes
        nodes_names_in_low_cluster = self.cdag.cluster_mapping[low_cluster.get_name()]
        nodes_names_in_high_cluster = self.cdag.cluster_mapping[high_cluster.get_name()]
        nodes_in_low_cluster = self.cdag.get_list_of_nodes_by_name(list_of_node_names= \
                                                               nodes_names_in_low_cluster, \
                                                                cg = self.cdag.cg)
        nodes_in_high_cluster = self.cdag.get_list_of_nodes_by_name(list_of_node_names= \
                                                               nodes_names_in_high_cluster, \
                                                                cg = self.cdag.cg)
        subgraph_cluster = CausalGraph(len(nodes_names_in_low_cluster) \
                                      + len(nodes_names_in_high_cluster), \
                                      nodes_names_in_low_cluster + nodes_names_in_high_cluster)
        subgraph_cluster.G = self.cdag.subgraph(nodes_in_low_cluster + nodes_in_high_cluster)

        # Define the local graph which contains possible separating sets, here it is 
        # low cluster union high cluster union low cluster parents 
        # high cluster is in low cluster parents per definition
        local_graph = self.cdag.get_local_graph(low_cluster)

        # Possibly replace subgraph_cluster and local_graph by index_arrays, 
        # as have to operate on entire adjacency matrix and data matrix
        cluster_node_indices = np.array([self.cdag.cg.G.node_map[node] \
                                    for node in subgraph_cluster.G.nodes])
        print(f'Cluster node indices are {cluster_node_indices}')
        local_graph_node_indices = np.array([self.cdag.cg.G.node_map[node] \
                                        for node in local_graph.G.nodes])
        print(f'Local graph node indices are {cluster_node_indices}')

        # Difference to local pc algorithm is that we consider only edges in the cluster
        # but as potential separating sets we consider cluster union cluster parents
        # Therefore stopping criterion is when separating set cardinality
        # exceed nonchilds in cluster

        # Skeleton discovery
        while self.cdag.max_nonchilds_of_cluster_nodes(low_cluster, local_graph) - 1 > depth:
            depth += 1
            edge_removal = []
            if self.show_progress:
                pbar.reset()
            for x in cluster_node_indices:
                if self.show_progress:
                    pbar.update()
                if self.show_progress:
                    pbar.set_description(f'Depth={depth}, working on node {x}')
                # Get all neighbors of node_x in the cluster, is integer values in adjacency matrix
                # x = subgraph_cluster.G.node_map[node_x] # x is index of node_x
                Neigh_x = self.cdag.cg.neighbors(x)
                print(f'Neigh_x is {Neigh_x}')
                # Remove neighbors that are not in cluster
                mask = np.isin(Neigh_x, cluster_node_indices)
                Neigh_x_in_clust = Neigh_x[mask]
                # Neigh_x_in_clust = np.delete(Neigh_x, np.where(\
                #     Neigh_x not in cluster_node_indices))
                print(f'Neig_x_in_Clust is {Neigh_x_in_clust}')
                possible_blocking_nodes = np.array(local_graph_node_indices)
                possible_blocking_nodes = np.delete(possible_blocking_nodes, \
                                                    np.where(possible_blocking_nodes == x))
                if len(Neigh_x) < depth - 1:
                    continue
                for y in Neigh_x_in_clust:
                    if self.verbose:
                        print('Testing edges from %d to %d' %(x, y))
                    # No other background functionality supported for now
                    # No parent checking for mns separation supported for now TODO
                    sepsets = set()
                    Neigh_x_no_y = np.delete(possible_blocking_nodes, \
                                             np.where(possible_blocking_nodes == y))
                    for S in combinations(Neigh_x_no_y, depth):
                        p = self.cdag.cg.ci_test(x, y, S)
                        if p > self.alpha:
                            if self.verbose:
                                print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                            if not self.stable:
                                edge1 = self.cdag.cg.G.get_edge(self.cdag.cg.G.nodes[x], \
                                                                self.cdag.cg.G.nodes[y])
                                if edge1 is not None:
                                    self.cdag.cg.G.remove_edge(edge1)
                                edge2 = self.cdag.cg.G.get_edge(self.cdag.cg.G.nodes[y], \
                                                                self.cdag.cg.G.nodes[x])
                                if edge2 is not None:
                                    self.cdag.cg.G.remove_edge(edge2)
                                append_value(self.cdag.cg.sepset, x, y, S)
                                append_value(self.cdag.cg.sepset, y, x, S)
                                break
                            else:
                                edge_removal.append((x, y))  # after all conditioning sets at
                                edge_removal.append((y, x))  # depth l have been considered
                                for s in S:
                                    sepsets.add(s)
                        else:
                            if self.verbose:
                                print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                    append_value(self.cdag.cg.sepset, x, y, tuple(sepsets))
                    append_value(self.cdag.cg.sepset, y, x, tuple(sepsets))
            if self.show_progress:
                pbar.refresh()

            for (x, y) in list(set(edge_removal)):
                edge1 = self.cdag.cg.G.get_edge(self.cdag.cg.G.nodes[x], \
                                                self.cdag.cg.G.nodes[y])
                if edge1 is not None:
                    self.cdag.cg.G.remove_edge(edge1)
                    print(f'Deleted edge from {x} to {y}')

        if self.show_progress:
            pbar.close()

        # TODO Orientation rules
        return self.cdag.cg

    def intra_cluster_phase(self, cluster):
        """
        Runs the intra-cluster phase of the ClustPC algorithm for a given cluster.
        Adapted from causallearn pc algorithm.
        Updates self.cdag.cg each step. 
        Input: cluster, a node object of CausalGraph
        """
        assert type(self.data) == np.ndarray
        assert 0 < self.alpha < 1
        
        no_of_var = self.data.shape[1]
        # Check if all variables are in the graph
        assert len(self.cdag.node_names) == no_of_var 

        # self.cdag.cg.set_ind_test(self.indep_test)

        depth = -1
        pbar = tqdm(total=no_of_var) if self.show_progress else None

        # Collect relevant nodes, i.e. cluster and parents of cluster in a list of Node objects
        # relevant_clusters, relevant_nodes = self.cdag.get_parent_plus(cluster)

        # Define the subgraph induced by the cluster nodes
        nodes_names_in_cluster = self.cdag.cluster_mapping[cluster.get_name()]
        nodes_in_cluster = self.cdag.get_list_of_nodes_by_name(list_of_node_names= \
                                                               nodes_names_in_cluster, \
                                                                cg = self.cdag.cg)
        subgraph_cluster = CausalGraph(len(nodes_names_in_cluster), nodes_names_in_cluster)
        subgraph_cluster.G = self.cdag.subgraph(nodes_in_cluster)

        # Define the local graph which contains possible separating sets
        local_graph = self.cdag.get_local_graph(cluster)

        # Possibly replace subgraph_cluster and local_graph by index_arrays, 
        # as have to operate on entire adjacency matrix and data matrix
        cluster_node_indices = np.array([self.cdag.cg.G.node_map[node] \
                                    for node in subgraph_cluster.G.nodes])
        print(f'Cluster node indices are {cluster_node_indices}')
        local_graph_node_indices = np.array([self.cdag.cg.G.node_map[node] \
                                        for node in local_graph.G.nodes])
        print(f'Local graph node indices are {cluster_node_indices}')

        # Difference to local pc algorithm is that we consider only edges in the cluster
        # but as potential separating sets we consider cluster union cluster parents
        # Therefore stopping criterion is when separating set cardinality
        # exceed nonchilds in cluster

        # Skeleton discovery
        while self.cdag.max_nonchilds_of_cluster_nodes(cluster, local_graph) - 1 > depth:
            depth += 1
            edge_removal = []
            if self.show_progress:
                pbar.reset()
            for x in cluster_node_indices:
                if self.show_progress:
                    pbar.update()
                if self.show_progress:
                    pbar.set_description(f'Depth={depth}, working on node {x}')
                # Get all neighbors of node_x in the cluster, is integer values in adjacency matrix
                # x = subgraph_cluster.G.node_map[node_x] # x is index of node_x
                Neigh_x = self.cdag.cg.neighbors(x)
                print(f'Neigh_x is {Neigh_x}')
                # Remove neighbors that are not in cluster
                mask = np.isin(Neigh_x, cluster_node_indices)
                Neigh_x_in_clust = Neigh_x[mask]
                # Neigh_x_in_clust = np.delete(Neigh_x, np.where(\
                #     Neigh_x not in cluster_node_indices))
                print(f'Neig_x_in_Clust is {Neigh_x_in_clust}')
                possible_blocking_nodes = np.array(local_graph_node_indices)
                possible_blocking_nodes = np.delete(possible_blocking_nodes, \
                                                    np.where(possible_blocking_nodes == x))
                if len(Neigh_x) < depth - 1:
                    continue
                for y in Neigh_x_in_clust:
                    if self.verbose:
                        print('Testing edges from %d to %d' %(x, y))
                    # No other background functionality supported for now
                    # No parent checking for mns separation supported for now TODO
                    sepsets = set()
                    Neigh_x_no_y = np.delete(possible_blocking_nodes, \
                                             np.where(possible_blocking_nodes == y))
                    for S in combinations(Neigh_x_no_y, depth):
                        p = self.cdag.cg.ci_test(x, y, S)
                        if p > self.alpha:
                            if self.verbose:
                                print('%d ind %d | %s with p-value %f\n' % (x, y, S, p))
                            if not self.stable:
                                edge1 = self.cdag.cg.G.get_edge(self.cdag.cg.G.nodes[x], \
                                                                self.cdag.cg.G.nodes[y])
                                if edge1 is not None:
                                    self.cdag.cg.G.remove_edge(edge1)
                                edge2 = self.cdag.cg.G.get_edge(self.cdag.cg.G.nodes[y], \
                                                                self.cdag.cg.G.nodes[x])
                                if edge2 is not None:
                                    self.cdag.cg.G.remove_edge(edge2)
                                append_value(self.cdag.cg.sepset, x, y, S)
                                append_value(self.cdag.cg.sepset, y, x, S)
                                break
                            else:
                                edge_removal.append((x, y))  # after all conditioning sets at
                                edge_removal.append((y, x))  # depth l have been considered
                                for s in S:
                                    sepsets.add(s)
                        else:
                            if self.verbose:
                                print('%d dep %d | %s with p-value %f\n' % (x, y, S, p))
                    append_value(self.cdag.cg.sepset, x, y, tuple(sepsets))
                    append_value(self.cdag.cg.sepset, y, x, tuple(sepsets))
            if self.show_progress:
                pbar.refresh()

            for (x, y) in list(set(edge_removal)):
                edge1 = self.cdag.cg.G.get_edge(self.cdag.cg.G.nodes[x], \
                                                self.cdag.cg.G.nodes[y])
                if edge1 is not None:
                    self.cdag.cg.G.remove_edge(edge1)
                    print(f'Deleted edge from {x} to {y}')

        if self.show_progress:
            pbar.close()

        # TODO Orientation rules
        return self.cdag.cg





            






