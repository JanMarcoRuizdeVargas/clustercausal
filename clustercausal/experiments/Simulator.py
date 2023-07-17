import causallearn
import castle
import numpy as np

from causallearn.graph.GraphClass import CausalGraph

from castle.datasets.simulator import IIDSimulation, DAG


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
        Run a simulation
        """
        if self.true_dag is None:
            self.true_dag = self.generate_dag()
        self.data = self.generate_data()
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
            n_clusters = np.random.randint(low=2, high=self.n_nodes / 2)
        if n_c_edges is None:
            n_c_edges = np.random.randint(
                low=n_clusters - 1, high=n_clusters * (n_clusters - 1) / 2
            )
        cluster_simulator = Simulator(
            n_nodes=n_clusters, n_edges=n_c_edges, dag_method=self.dag_method
        )
        cluster_graph, _ = cluster_simulator.run()

        cluster_names = [f"C{i+1}" for i in range(n_clusters)]
        for node_name in self.true_dag.node_names:
            pass
        # TODO finish this
        return cluster_graph
