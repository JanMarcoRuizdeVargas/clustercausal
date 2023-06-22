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
from causallearn.utils.cit import *
from causallearn.utils.PCUtils import Helper, Meek, SkeletonDiscovery, UCSepset
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import \
    orient_by_background_knowledge
from causallearn.search.ConstraintBased import pc, pc_alg

from clustcausal.clusterdag.cluster_dag import CDAG

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ClustPC():
    """
    Runs the ClustPC algorithm according to my master thesis.
    Adapted from causallearn pc algorithm. 
    
    """
    def __init__(self, cdag : CDAG):#, data: ndarray):
        self.cdag = cdag
        self.cdag.cdag_to_mpdag() # Get pruned MPDAG from CDAG
        self.cdag.get_topological_ordering()  # Get topological ordering of CDAG
        # self.data = data
        # if data is not pd.DataFrame:
        #     logging.info('Data converted into pandas dataframe with columns X1, X2, ...')
        #     data = pd.DataFrame(data)
        #     data.columns = self.cdag.node_names

    def set_parameters(self,
        cdag: CDAG,
        data: ndarray,
        node_names: List[str] | None,
        alpha: float,
        indep_test: str,
        stable: bool,
        uc_rule: int,
        uc_priority: int,
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
        self.node_names = node_names
        self.alpha = alpha
        self.indep_test = indep_test
        self.stable = stable
        self.uc_rule = uc_rule
        self.uc_priority = uc_priority
        self.background_knowledge = background_knowledge
        self.verbose = verbose
        self.show_progress = show_progress
        self.kwargs = kwargs
                       

    def run(self) -> CausalGraph:
        for cluster in self.cdag.cdag_list_of_topological_sort:
            self.inter_cluster_phase(cluster)
            # Apply Meek edge orientation rules
            self.intra_cluster_phase(cluster)
            # Apply Meek edge orientation rules
        # Apply Meek edge orientation rules
        return self.cdag.cg # Return CausalGraph of the CDAG

    def inter_cluster_phase(self, cluster):
        """
        Runs the inter-cluster phase of the ClustPC algorithm for a given cluster.
        cluster is a node object of CausalGraph
        """
        # Identify the nodes needed to be considered, i.e. cluster and parents of cluster
        parent_clusters = self.cdag.cluster_dag.G.get_parents(cluster)
        self.cdag.cg.G.get_parents

    def intra_cluster_phase(self, cluster):
        """
        Runs the intra-cluster phase of the ClustPC algorithm for a given cluster.
        Adapted from causallearn pc algorithm.
        Updates self.cdag.cg each step. 
        """
        assert type(self.data) == np.ndarray
        assert 0 < self.alpha < 1
        
        no_of_var = self.data.shape[1]
        # Check if all variables are in the graph
        assert len(self.cdag.cg.G.get_nodes) == no_of_var 

        self.cdag.cg.set_ind_test(self.indep_test)

        depth = -1
        pbar = tqdm(total=no_of_var) if self.show_progress else None
        # Collect relevant nodes, i.e. cluster and parents of cluster in a list of Node objects
        relevant_clusters = self.cdag.cluster_graph.G.get_parents(cluster)
        for i in range(len(relevant_clusters)):
            relevant_clusters[i] = relevant_clusters[i].get_name()
        relevant_nodes = []
        for cluster in relevant_clusters:
            relevant_nodes.extend(self.cdag.cluster_mapping[cluster])
        for i in relevant_nodes:
            relevant_nodes[i] = ClustPC.get_key_by_value(self.cdag.cg.G.node_map, i)
        
        # Define the local graph on which to run the intra cluster phase, restrict data
        local_data = self.data[:,] # TODO have to figure out how to restrict array correctly

    @staticmethod
    def get_key_by_value(dictionary, value):
        # Helper function to get Node object from node_map value i regarding GraphNode object
        for key, val in dictionary.items():
            if val == value:
                return key
        return None  # Value not found in the dictionary

