import causallearn
import castle

from causallearn.graph.GraphClass import CausalGraph

from castle.datasets.simulator import IIDSimulation, DAG


class Simulator:
    """
    A simulator to generate a causal graph and data from it.
    """

    def __init__(self):
        """
        Initialize an instance
        """
        pass

    def run(
        self,
        true_dag,
        n_nodes,
        n_edges,
        dag_method,
        weight_range,
        distribution_type,
        scm_method,
        sample_size,
        seed,
        noise_scale=1.0,
    ):
        """
        Run a simulation
        """
        if true_dag is None:
            true_dag = self.generate_dag(
                n_nodes, n_edges, dag_method, weight_range, seed
            )
        data = self.generate_data(
            true_dag, sample_size, scm_method, distribution_type, noise_scale
        )
        return true_dag, data

    def generate_dag(
        self,
        n_nodes,
        n_edges,
        dag_method="erdos_renyi",
        weight_range=(-1, 2),
        seed=42,
    ):
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
        if dag_method == "erdos_renyi":
            W = DAG.erdos_renyi(
                n_nodes, n_edges, weight_range=weight_range, seed=seed
            )
        elif dag_method == "scale_free":
            W = DAG.scale_free(
                n_nodes, n_edges, weight_range=weight_range, seed=seed
            )
        elif dag_method == "bipartite":
            W = DAG.bipartite(
                n_nodes, n_edges, weight_range=weight_range, seed=seed
            )
        elif dag_method == "hierarchical":
            W = DAG.hierarchical(
                n_nodes, n_edges, weight_range=weight_range, seed=seed
            )
        # elif dag_method == "low_rank":
        #     W = DAG.low_rank(
        #         n_nodes, n_edges, weight_range=weight_range, seed=seed
        #     )
        # for weighted adjacency matrix W create CausalGraph object
        true_dag = CausalGraph(no_of_var=W.shape[0])
        true_dag.adjacency_matrix = W
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if W[i, j] != 0:
                    # Make tail at i and arrow at j
                    true_dag.G.graph[i, j] = -1
                    true_dag.G.graph[j, i] = 1

        return true_dag

    def generate_data(
        self, true_dag, sample_size, scm_method, distribution_type, noise_scale
    ):
        """
        Generate data from the causal graph
        Arguments:
            true_dag: the causal graph
            distribution_type: distribution type of the data
            sample_size: size of the data
        Output:
            data: sample_size x no_of_nodes ndarray
        """
        if true_dag.adjacency_matrix is None:
            raise ValueError("Adjacency matrix is None")
        dataset = IIDSimulation(
            true_dag.adjacency_matrix,
            n=sample_size,
            method=scm_method,
            sem_type=distribution_type,
            noise_scale=noise_scale,
        )
        return dataset.X
