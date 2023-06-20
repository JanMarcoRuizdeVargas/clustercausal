import numpy as np
import pandas as pd
import networkx as nx
import causallearn
import castle
import pydot
import logging

from itertools import combinations
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class CDAG:
    """
    Class for functionality regarding CDAGS

    attributes:
        clusters: dictionary of clusters
        cluster_edges: list of tuples of cluster edges
        graph: CausalGraph object
        background_knowledge: BackgroundKnowledge object
        node_names: list of node names
        node_indices: dictionary that points to which cluster the node is in
    
    methods:
        cdag_to_mpdag: constructs a MPDAG from a CDAG

    """
    def __init__(self, cluster_mapping: dict, 
        cluster_edges: list):
        """
        Construct a CDAG object from a cluster dictionary
        The CDAG is stored as a dictionary. 
        The cluster_nodes are stored as a dictionary pointing 
        to a list of cluster members. 
        The cluster_edges are stored as a list of tuples.
        An example CDAG:
            cdag = {}
            cdag['clusters'] = {'C1':['X1','X2','X3'], 'C2': ['X4','X5']}
            cdag['cluster_edges'] = [('C1','C2')] 
        """
        self.clusters = cluster_mapping
        self.cluster_edges = cluster_edges
        self.node_names = []
        for cluster in self.clusters:
            self.node_names.extend(self.clusters[cluster])
        self.node_indices = {} # Dictionary that points to which cluster the node is in
        for node in self.node_names:
            for cluster, vertice in self.clusters.items():
                if node in vertice:
                    self.node_indices[node] = cluster
        

    def cdag_to_mpdag(self):
        """
        Constructs a MPDAG from a CDAG and stores it in a causallearn
        BackgroundKnowledge object. 
        """
        # Create the list of node_names needed for CausalGraph
        self.cg = CausalGraph(no_of_var = len(self.node_names), 
                                                      node_names = self.node_names)
        # Remove edges that are forbidden by the CDAG
        # self.background_knowledge = BackgroundKnowledge()
        for edge in self.cg.G.get_graph_edges():
            # There must be a better way to do this by only adressing the edges needed to be changed
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            cluster1 = self.node_indices[node1.get_name()]
            cluster2 = self.node_indices[node2.get_name()]
            if cluster1 != cluster2:
                if (cluster1, cluster2) not in self.cluster_edges:
                    self.cg.G.remove_edge(edge)
                    logging.info(f'removed edge: ({node1.get_name()},{node2.get_name()})')
                if (cluster1, cluster2) in self.cluster_edges:
                    self.cg.G.remove_edge(edge)
                    self.cg.G.add_directed_edge(node1, node2)
                    logging.info(f'oriented edge: ({node1.get_name()},{node2.get_name()})')

    def cdag_from_background_knowledge(self):
        """
        Construct a CDAG object from background knowledge
        Types of background knowledge:
            Todo
            -required edges
            -forbidden edges
            -required ancestors
            -forbidden ancestors
        """
        pass

    def background_knowledge_from_cdag(self):
        """
        Construct background knowledge from a CDAG object
        Types of background knowledge:
            Todo
            -required edges
            -forbidden edges
            -required ancestors
            -forbidden ancestors
        """
        pass