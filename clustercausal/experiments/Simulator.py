import causallearn
import castle

from castle.datasets.simulator import IIDSimulation, DAG


class Simulator:
    """
    A simulator to generate a causal graph and data from it.
    """

    def __init__(
        self,
        true_dag,
        no_of_nodes,
        no_of_edges,
        distribution_type,
        sample_size,
        seed,
    ):
        """
        Initialize number of nodes and edges in the causal graph
        """
        if true_dag is None:
            true_dag = self.generate_dag(no_of_nodes, no_of_edges, seed)
        data = self.generate_data(true_dag, distribution_type, sample_size)
        return true_dag, data

    def generate_dag(
        self,
        n_nodes,
        n_edges,
        method="erdos_renyi",
        weight_range=(-1, 2),
        seed=42,
    ):
        """
        Generate a random DAG with gcastle
        Arguments:
            n_nodes: number of nodes in the causal graph
            n_edges: number of edges in the causal graph
            method: method to generate the causal graph]
                    methods supported: erdos_renyi, scale_free, bipartite, hierarchical, low_rank
            seed: seed for the random number generator
        Output:
            A CausalGraph object
        """
        if method == "erdos_renyi":
            W = DAG.erdos_renyi(
                n_nodes, n_edges, weight_range=weight_range, seed=seed
            )
        elif method == "scale_free":
            W = DAG.scale_free(
                n_nodes, n_edges, weight_range=weight_range, seed=seed
            )
        elif method == "bipartite":
            W = DAG.bipartite(
                n_nodes, n_edges, weight_range=weight_range, seed=seed
            )
        elif method == "hierarchical":
            W = DAG.hierarchical(
                n_nodes, n_edges, weight_range=weight_range, seed=seed
            )
        elif method == "low_rank":
            W = DAG.low_rank(
                n_nodes, n_edges, weight_range=weight_range, seed=seed
            )
        # for weighted adjacency matrix W create CausalGraph object
        # TODO
        # For getting CPDAG/PAG use causallearn
        return dag

    def generate_data(self, true_dag, distribution_type, sample_size):
        """
        Generate data from the causal graph
        Arguments:
            true_dag: the causal graph
            distribution_type: distribution type of the data
            sample_size: size of the data
        Output:
            data: sample_size x no_of_nodes ndarray
        """
        data = None
        return data
