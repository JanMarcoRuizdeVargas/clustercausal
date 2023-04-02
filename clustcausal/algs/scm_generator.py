import networkx as nx
import numpy as np
import logging

class SCMGenerator():
    """
    SCM generator class
    methods:
        __init__: initializes the SCM generator with the specified number of nodes, clusters and a sparsity parameter
        make_dag: generates a random DAG with the specified number of nodes and clusters
        scm_equations: returns the SCM equations in written form
        generate_data: generates data according to the SCM equations
    """

    def __init__(self, num_nodes, num_clusters, in_cluster_connectivity = 0.5, between_cluster_connectivity = 0.2, seed = None):
        """
        Initialize the SCM generator
        args:
            num_nodes: number of nodes in the graph
            num_clusters: number of clusters in the graph
            in_cluster_connectivity: number between 0 and 1, the probability of getting an edge within a cluster
            between_cluster_connectivity: number between 0 and 1, the probability of getting an edge outside of clusters

        """
        self.seed = seed
        self.nodes = list(range(num_nodes))
        self.num_clusters = num_clusters
        self.cluster_assignment = []
        np.random.seed(self.seed)
        cluster_probabilities = np.random.uniform(low = 0.5, high = 1, size = num_clusters)
        self.cluster_probabilities = cluster_probabilities / np.sum(cluster_probabilities)
        # logging.info(f'Cluster probabilities: {self.cluster_probabilities}')
        for node in self.nodes:
            # np.random.seed(self.seed)
            self.cluster_assignment += [np.random.choice(self.num_clusters, p = self.cluster_probabilities)]
        # logging.info(f'Cluster assignment: {self.cluster_assignment}')
        self.cluster_sizes = []
        for cluster in range(num_clusters):
            self.cluster_sizes += [np.sum(np.array(self.cluster_assignment) == cluster)]
        # logging.info(f'Cluster sizes: {self.cluster_sizes}')
        self.in_cluster_connectivity = in_cluster_connectivity
        self.between_cluster_connectivity = between_cluster_connectivity
        self.graph = None
        self.edge_weights = {}
        self.clusters = {}
        for cluster in range(self.num_clusters):
            self.clusters[cluster] = (self.cluster_sizes[cluster], np.where(np.array(self.cluster_assignment) == cluster))

    def make_dag(self, seed = None, directed = True):
        '''
        Creates a DAG with the nx package
        '''
        # np.random.seed(self.seed)
        acyclicity = False
        while acyclicity == False:
            self.graph = nx.random_partition_graph(sizes = self.cluster_sizes, p_in = self.in_cluster_connectivity, p_out = self.between_cluster_connectivity, seed = self.seed, directed = directed)
            for (u,v) in self.graph.edges:
                if ((u,v) in self.graph.edges()) and ((v,u) in self.graph.edges()):
                    self.graph.remove_edge(v,u)
            for (u, v) in self.graph.edges:
                # np.random.seed(self.seed)
                self.edge_weights[(u, v)] = np.round(np.random.uniform(low = -2, high = 2), decimals =2 )
            self.edge_list = list(self.graph.edges)
            self.parents = {}
            for node in self.nodes:
                self.parents[node] = []
            for edge in list(self.graph.edges):
                self.parents[edge[1]] += [edge[0]]
            acyclicity = nx.is_directed_acyclic_graph(self.graph)

    
    def scm_equations(self):
        """
        Returns the SCM equations in written form
        """
        equations = {}
        for node in self.nodes:
            equations[node] = f'X{node} = '
            for parent in self.graph.predecessors(node):
                equations[node] += f'{self.edge_weights[(parent, node)]} * X{parent} + '
        for node in self.nodes:
            equations[node] += f'N{node}'
        self.written_equations = equations
        # self.scm_functions = {}
        self.topological_order = list(nx.topological_sort(self.graph))

        # self.func_list = []
        # for node in self.topological_order:
        #     def create_function(*args):
        #         def inner(*args):
        #             for pa in args:
        #                 value += self.edge_weights[(pa, node)] * pa
        #             return value
        #         return inner
        #     self.scm_functions[node] = f(self.parents[node])

    
    def generate_data(self, samples = 1000):
        """
        Generate data according to the scm equations
        """
        data = np.zeros((len(self.nodes), samples))
        for node in self.topological_order:
            data_node = np.random.normal(size = samples)
            for parent in self.parents[node]:
                data_node += self.edge_weights[(parent, node)] * data[parent,:]
            data[node,:] = data_node
        self.data = data
