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
                       

    def run(self):
        for cluster in self.cdag.cdag_list_of_topological_sort:
            pass
            

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
        """
        pass