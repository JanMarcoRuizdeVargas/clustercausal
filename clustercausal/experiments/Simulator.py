import causallearn
import castle
import numpy as np
import itertools
import networkx as nx

from causallearn.graph.GraphClass import CausalGraph
from causallearn.graph.Endpoint import Endpoint

from castle.datasets.simulator import IIDSimulation, DAG

from clustercausal.clusterdag.ClusterDAG import ClusterDAG


class Simulator:
    """
    A simulator to generate a causal graph and data from it.
    Arguments:
        true_dag: the true causal graph(needs weighted_adjacency_matrix)
        n_nodes: number of nodes in the true_dag if simulated
        n_edges: number of edges in the true_dag if simulated
        dag_method: method to generate the causal graph,
                    methods supported: erdos_renyi, scale_free, bipartite, hierarchical
                    not supported: low_rank
        n_clusters: number of clusters in the cluster graph, if None then random
        n_c_edges: number of edges in the cluster graph, if None then random
        weight_range: range of weights of adjacency matrix in the causal graph
        distribution_type: distribution type of the data
                    methods supported:
                        gauss, exp, gumbel, uniform, logistic (linear);
                        mlp, mim, gp, gp-add, quadratic (nonlinear).
        scm_method: linear or nonlinear, default is linear
        sample_size: size of the data
        seed: seed for the random number generator
        noise_scale: scale of the noise in the data
    """

    def __init__(
        self,
        true_dag=None,
        n_nodes=7,
        n_edges=13,
        dag_method="erdos_renyi",
        n_clusters=None,
        # n_c_edges=None,
        weight_range=(-1, 2),
        distribution_type="gauss",
        scm_method="linear",
        sample_size=10000,
        seed=42,
        noise_scale=1.0,
    ):
        """
        Initialize an instance, set parameters
        """
        self.true_dag = true_dag
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        print("Warning: n_edges is not exact due to gcastle implementation")
        self.dag_method = dag_method
        self.n_clusters = n_clusters
        # self.n_c_edges = n_c_edges
        self.weight_range = weight_range
        self.distribution_type = distribution_type
        self.scm_method = scm_method
        self.sample_size = sample_size
        self.seed = seed
        self.noise_scale = noise_scale

    def run(self) -> ClusterDAG:
        """
        Run the simulator and generate a cluster_dag
        Returns:
            cluster_dag: a ClusterDAG object
                with true_dag, data, cluster_graph, cluster_mapping attributes
        """
        dag = self.true_dag
        if dag is None:
            dag = self.generate_dag(
                self.n_nodes,
                self.n_edges,
                self.dag_method,
                self.weight_range,
                self.seed,
            )
        cluster_dag = self.generate_clustering(dag, self.n_clusters, self.seed)

        # adj_dag, cluster_graph, cluster_mapping = self.generate_clustering(
        #     dag, self.n_clusters, self.n_c_edges, self.dag_method, self.seed
        # )
        data = self.generate_data(
            cluster_dag.true_dag,
            self.sample_size,
            self.distribution_type,
            self.scm_method,
            self.noise_scale,
        )

        # cluster_edges = []
        # for edge in cluster_graph.G.get_graph_edges():
        #     node1_name = edge.get_node1().get_name()
        #     node2_name = edge.get_node2().get_name()
        #     cluster_edges.append((node1_name, node2_name))

        # Ensure that node names in true_dag and est_dag (calculated later)
        # are the same
        # node_names = [node.get_name() for node in adj_dag.G.get_nodes()]
        # cluster_dag = ClusterDAG(cluster_mapping, cluster_edges, node_names)
        # cluster_dag.true_dag = adj_dag

        cluster_dag.data = data
        return cluster_dag

    @staticmethod
    def generate_dag(
        n_nodes, n_edges, dag_method, weight_range, seed, node_names=None
    ) -> CausalGraph:
        """
        Generate a random DAG with gcastle
        Arguments:
            n_nodes: number of nodes in the causal graph
            n_edges: number of edges in the causal graph
            dag_method: method to generate the causal graph]
                    methods supported: erdos_renyi, scale_free, bipartite, hierarchical
                    not supported: low_rank
            seed: seed for the random number generator
        Output:
            A CausalGraph object
        """
        if dag_method == "erdos_renyi":
            W = DAG.erdos_renyi(
                n_nodes,
                n_edges,
                weight_range=weight_range,
                seed=seed,
            )
        elif dag_method == "scale_free":
            W = DAG.scale_free(
                n_nodes,
                n_edges,
                weight_range=weight_range,
                seed=seed,
            )
        elif dag_method == "bipartite":
            W = DAG.bipartite(
                n_nodes,
                n_edges,
                weight_range=weight_range,
                seed=seed,
            )
        elif dag_method == "hierarchical":
            W = DAG.hierarchical(
                n_nodes,
                n_edges,
                weight_range=weight_range,
                seed=seed,
            )
        # elif dag_method == "low_rank":
        #     W = DAG.low_rank(
        #         n_nodes, n_edges, weight_range=weight_range, seed=seed
        #     )
        # for weighted adjacency matrix W create CausalGraph object
        dag = CausalGraph(no_of_var=W.shape[0], node_names=node_names)
        dag.G.graph = np.zeros((W.shape[0], W.shape[1]))
        dag.weighted_adjacency_matrix = W
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if W[i, j] != 0:
                    # Make tail at i and arrow at j
                    dag.G.graph[i, j] = -1
                    dag.G.graph[j, i] = 1
        return dag

    @staticmethod
    def generate_data(
        dag: CausalGraph,
        sample_size,
        distribution_type,
        scm_method,
        noise_scale,
    ):
        """
        Generate data from the causal graph
        Arguments:
            dag: the causal graph
            sample_size: size of the data
            distribution_type: distribution type of the data
                    methods supported:
                        gauss, exp, gumbel, uniform, logistic (linear);
                        mlp, mim, gp, gp-add, quadratic (nonlinear).
            scm_method: linear or nonlinear, default is linear
            noise_scale: scale of the noise in the data
        Output:
            data: sample_size x no_of_nodes ndarray
        """
        if dag.weighted_adjacency_matrix is None:
            raise ValueError("Adjacency matrix is None")
        dataset = IIDSimulation(
            dag.weighted_adjacency_matrix,
            n=sample_size,
            method=scm_method,
            sem_type=distribution_type,
            noise_scale=noise_scale,
        )
        return dataset.X

    @staticmethod
    def generate_clustering(dag: CausalGraph, n_clusters, seed):
        """
        Generate an admissible (no cycles) clustering from dag
        Arguments:
            dag: the causal graph
            n_clusters: number of clusters in the cluster graph, if None then random
            n_c_edges: number of edges in the cluster graph, if None then random
            dag_method: method to generate the causal graph
                    methods supported: erdos_renyi, scale_free, bipartite, hierarchical
                    not supported: low_rank
            seed: seed for the random number generator
        Output:
            cluster_dag: a ClusterDAG object
        Adjusts true_dag such that cluster_graph is admissible
        """

        # Generate a cluster graph
        n_nodes = dag.G.graph.shape[0]
        np.random.seed(seed)
        node_names = [node.get_name() for node in dag.G.get_nodes()]

        if n_clusters is None:
            n_clusters = np.random.randint(
                low=2, high=int(np.ceil(n_nodes / 2)) + 1
            )

        # Get topological ordering of nodes, based on that generate admissible cluster graph
        nx_helper_graph = nx.DiGraph()
        edge_name_list = []
        for edge in dag.G.get_graph_edges():
            node1_name = edge.get_node1().get_name()
            node2_name = edge.get_node2().get_name()
            edge_name_list.append((node1_name, node2_name))
        nx_helper_graph.add_edges_from(edge_name_list)
        nx_helper_graph.add_nodes_from(
            node_names
        )  # ensure that all nodes are in the graph
        topological_ordering = list(nx.topological_sort(nx_helper_graph))
        # successively partition the topological ordering into clusters
        # Each cluster gets at least one node
        # Get cluster cutoffs by drawing without replacement from topological ordering
        node_range = list(range(1, n_nodes))
        cluster_cutoffs = sorted(
            np.random.choice(node_range, size=n_clusters - 1, replace=False)
        )
        cluster_cutoffs.append(
            n_nodes
        )  # ensure that last cluster has all remaining nodes
        cluster_mapping = {}
        cluster_names = [f"C{i+1}" for i in range(n_clusters)]
        j = 0
        l = 0
        u = cluster_cutoffs[0]
        for cluster in cluster_names:
            cluster_mapping[cluster] = []
            for i in range(l, u):
                cluster_mapping[cluster].append(topological_ordering[i])
            l = u
            if j < n_clusters - 1:
                u = cluster_cutoffs[j + 1]
                j += 1

        # Cluster edges
        cluster_edges = []
        for edge in dag.G.get_graph_edges():
            node1_name = edge.get_node1().get_name()
            node2_name = edge.get_node2().get_name()
            c1_name = ClusterDAG.find_key(cluster_mapping, node1_name)
            c2_name = ClusterDAG.find_key(cluster_mapping, node2_name)
            if c1_name != c2_name:
                endpoint1 = edge.get_endpoint1()
                endpoint2 = edge.get_endpoint2()
                if endpoint1 == Endpoint.ARROW and endpoint2 == Endpoint.TAIL:
                    if (c2_name, c1_name) not in cluster_edges:
                        cluster_edges.append((c2_name, c1_name))
                if endpoint1 == Endpoint.TAIL and endpoint2 == Endpoint.ARROW:
                    if (c1_name, c2_name) not in cluster_edges:
                        cluster_edges.append((c1_name, c2_name))
                # if (endpoint1 == Endpoint.TAIL_AND_ARROW or endpoint1 == Endpoint.ARROW_AND_ARROW) \
                #         and (endpoint2 == Endpoint.TAIL_AND_ARROW or endpoint2 == Endpoint.ARROW_AND_ARROW):
                #     cluster_edges.append((c1_name, c2_name))
                #     cluster_edges.append((c2_name, c1_name)) # Later for confounders

        cluster_dag = ClusterDAG(
            cluster_mapping, cluster_edges, node_names=node_names
        )
        cluster_dag.true_dag = dag
        return cluster_dag

        # # OLD CODE
        """
        # if n_clusters is None:
        #     n_clusters = np.random.randint(
        #         low=3, high=int(np.ceil(n_nodes / 2 + 1)) + 1
        #     )
        # if n_c_edges is None:
        #     n_c_edges = np.random.randint(
        #         low=n_clusters - 1,
        #         high=int(np.ceil(n_clusters * (n_clusters - 1) / 2)) + 1,
        #     )
        # weight_range = (1, 1)
        # cluster_names = [f"C{i+1}" for i in range(n_clusters)]
        # cluster_graph = Simulator.generate_dag(
        #     n_clusters,
        #     n_c_edges,
        #     dag_method,
        #     weight_range,
        #     seed,
        #     node_names=cluster_names,
        # )
        # cluster_graph.node_names = cluster_names

        # # Give each cluster a probability of being chosen
        # cluster_probabilities = np.random.dirichlet(
        #     np.ones(n_clusters), size=1
        # )[0]

        # # Make cluster mapping, ensure that each cluster has at least one node
        # cluster_mapping = {}
        # dag.node_names = ClusterDAG.get_node_names_from_list(dag.G.nodes)
        # nodes_left_to_assign = dag.node_names.copy()
        # for cluster_name in cluster_graph.node_names:
        #     first_node = np.random.choice(nodes_left_to_assign)
        #     nodes_left_to_assign.remove(first_node)
        #     cluster_mapping[cluster_name] = [first_node]
        # for node_name in nodes_left_to_assign:
        #     chosen_cluster = np.random.choice(
        #         cluster_graph.node_names, p=cluster_probabilities
        #     )
        #     cluster_mapping[chosen_cluster] += [node_name]

        # # Adjust true_dag such that cluster_graph is admissible
        # for cluster1, cluster2 in itertools.combinations(
        #     cluster_graph.node_names, 2
        # ):
        #     for node1 in cluster_mapping[cluster1]:
        #         for node2 in cluster_mapping[cluster2]:
        #             c1 = ClusterDAG.get_node_by_name(cluster1, cluster_graph)
        #             c2 = ClusterDAG.get_node_by_name(cluster2, cluster_graph)
        #             n1 = ClusterDAG.get_node_by_name(node1, dag)
        #             n2 = ClusterDAG.get_node_by_name(node2, dag)
        #             c1_indice = cluster_graph.G.node_map[c1]
        #             c2_indice = cluster_graph.G.node_map[c2]
        #             n1_indice = dag.G.node_map[n1]
        #             n2_indice = dag.G.node_map[n2]
        #             # Reorient edge n1 -> n2 to n1 <- n2 if c1 <- c2
        #             if (
        #                 cluster_graph.G.graph[c1_indice, c2_indice] == 1
        #                 and cluster_graph.G.graph[c2_indice, c1_indice] == -1
        #                 and dag.G.graph[n1_indice, n2_indice] == -1
        #                 and dag.G.graph[n2_indice, n1_indice] == 1
        #             ):
        #                 dag.G.graph[n1_indice, n2_indice] = 1
        #                 dag.G.graph[n2_indice, n1_indice] = -1
        """

        # return dag, cluster_graph, cluster_mapping


# simulation = Simulator(n_nodes=5, n_edges=7, seed=1234)
# cluster_dag = simulation.run()
