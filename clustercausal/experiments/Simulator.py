import causallearn
import castle
import numpy as np
import itertools

from causallearn.graph.GraphClass import CausalGraph

from castle.datasets.simulator import IIDSimulation, DAG

from clustercausal.clusterdag.ClusterDAG import ClusterDAG


class Simulator:
    """
    A simulator to generate a causal graph and data from it.
    """

    def __init__(
        self,
        true_dag=None,
        n_nodes=7,
        n_edges=13,
        dag_method="erdos_renyi",
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
        self.dag_method = dag_method
        self.weight_range = weight_range
        self.distribution_type = distribution_type
        self.scm_method = scm_method
        self.sample_size = sample_size
        self.seed = seed
        self.noise_scale = noise_scale

    def run(self):
        """
        Run the simulator and get cluster_graph, true_dag,
        cluster_mapping and data
        """
        self.true_dag, self.data = self.create_dag_and_data()
        cluster_graph, self.true_dag, cluster_mapping = self.generate_clustering(
            n_clusters=None, n_c_edges=None
        )
        self.cluster_graph = cluster_graph
        self.cluster_mapping = cluster_mapping

        cluster_edges = self.cluster_graph.G.get_graph_edges()
        cluster_edges_names = []
        for edge in cluster_edges:
            node1_name = edge.get_node1().get_name()
            node2_name = edge.get_node2().get_name()
            cluster_edges_names.append((node1_name, node2_name))
        # Create a ClusterDAG object
        simulated_cluster_dag = ClusterDAG(self.cluster_mapping, cluster_edges_names)
        simulated_cluster_dag.true_dag = self.true_dag
        simulated_cluster_dag.data = self.data
        return simulated_cluster_dag

    def create_dag_and_data(self):
        """
        Create dag and data
        """
        if self.true_dag is None:
            self.true_dag = self.generate_dag()
        self.data = self.generate_data()
        # TODO put cluster_graph in here too
        return self.true_dag, self.data

    def generate_dag(self):
        """
        Generate a random DAG with gcastle
        Arguments:
            n_nodes: number of nodes in the causal graph
            n_edges: number of edges in the causal graph
            method: method to generate the causal graph]
                    methods supported: erdos_renyi, scale_free, bipartite, hierarchical
                    not supported: low_rank
            seed: seed for the random number generator
        Output:
            A CausalGraph object
        """
        if self.dag_method == "erdos_renyi":
            W = DAG.erdos_renyi(
                self.n_nodes,
                self.n_edges,
                weight_range=self.weight_range,
                seed=self.seed,
            )
        elif self.dag_method == "scale_free":
            W = DAG.scale_free(
                self.n_nodes,
                self.n_edges,
                weight_range=self.weight_range,
                seed=self.seed,
            )
        elif self.dag_method == "bipartite":
            W = DAG.bipartite(
                self.n_nodes,
                self.n_edges,
                weight_range=self.weight_range,
                seed=self.seed,
            )
        elif self.dag_method == "hierarchical":
            W = DAG.hierarchical(
                self.n_nodes,
                self.n_edges,
                weight_range=self.weight_range,
                seed=self.seed,
            )
        # elif dag_method == "low_rank":
        #     W = DAG.low_rank(
        #         n_nodes, n_edges, weight_range=weight_range, seed=seed
        #     )
        # for weighted adjacency matrix W create CausalGraph object
        true_dag = CausalGraph(no_of_var=W.shape[0])
        true_dag.G.graph = np.zeros((W.shape[0], W.shape[1]))
        true_dag.weighted_adjacency_matrix = W
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if W[i, j] != 0:
                    # Make tail at i and arrow at j
                    true_dag.G.graph[i, j] = -1
                    true_dag.G.graph[j, i] = 1
        return true_dag

    def generate_data(self):
        """
        Generate data from the causal graph
        Arguments:
            true_dag: the causal graph
            distribution_type: distribution type of the data
            sample_size: size of the data
        Output:
            data: sample_size x no_of_nodes ndarray
        """
        if self.true_dag.weighted_adjacency_matrix is None:
            raise ValueError("Adjacency matrix is None")
        dataset = IIDSimulation(
            self.true_dag.weighted_adjacency_matrix,
            n=self.sample_size,
            method=self.scm_method,
            sem_type=self.distribution_type,
            noise_scale=self.noise_scale,
        )
        return dataset.X

    def generate_clustering(self, n_clusters, n_c_edges):
        """
        Generate an admissible (no cycles)
        clustering from self.true_dag
        """
        np.random.seed(self.seed)
        if n_clusters is None:
            n_clusters = np.random.randint(low=3, high=self.n_nodes / 2 + 1)
        if n_c_edges is None:
            n_c_edges = np.random.randint(
                low=n_clusters - 1, high=n_clusters * (n_clusters - 1) / 2
            )

        cluster_simulator = Simulator(
            n_nodes=n_clusters, n_edges=n_c_edges, dag_method=self.dag_method
        )
        cluster_graph, _ = cluster_simulator.create_dag_and_data()
        cluster_names = [f"C{i+1}" for i in range(n_clusters)]
        cluster_graph.nodes_names = cluster_names

        # Give each cluster a probability of being chosen
        cluster_probabilities = np.random.dirichlet(np.ones(n_clusters), size=1)[0]
        cluster_mapping = {}
        # Make cluster mapping
        for cluster_name in cluster_names:
            cluster_mapping[cluster_name] = []
        for node_name in self.true_dag.node_names:
            chosen_cluster = np.random.choice(cluster_names, p=cluster_probabilities)
            cluster_mapping[chosen_cluster] += [node_name]
        # Adjust true_dag such that cluster_graph is admissible
        for cluster1, cluster2 in itertools.combinations(cluster_names, 2):
            for node1 in cluster_mapping[cluster1]:
                for node2 in cluster_mapping[cluster2]:
                    c1 = ClusterDAG.get_node_by_name(cluster1, cluster_graph)
                    c2 = ClusterDAG.get_node_by_name(cluster2, cluster_graph)
                    n1 = ClusterDAG.get_node_by_name(node1, self.true_dag)
                    n2 = ClusterDAG.get_node_by_name(node2, self.true_dag)
                    c1_indice = cluster_graph.G.node_map[c1]
                    c2_indice = cluster_graph.G.node_map[c2]
                    n1_indice = self.true_dag.G.node_map[n1]
                    n2_indice = self.true_dag.G.node_map[n2]
                    # Reorient edge n1 -> n2 to n1 <- n2 if c1 <- c2
                    if (
                        cluster_graph.G.graph[c1_indice, c2_indice] == 1
                        and cluster_graph.G.graph[c2_indice, c1_indice] == -1
                        and self.true_dag.G.graph[n1_indice, n2_indice] == -1
                        and self.true_dag.G.graph[n2_indice, n1_indice] == 1
                    ):
                        self.true_dag.G.graph[n1_indice, n2_indice] = 1
                        self.true_dag.G.graph[n2_indice, n1_indice] = -1

        return cluster_graph, self.true_dag, cluster_mapping


simulation = Simulator()
simulated_cluster_dag = simulation.run()
