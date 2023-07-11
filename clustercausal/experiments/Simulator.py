import causallearn
import gCastle


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

    def generate_dag(self, no_of_nodes, no_of_edges, seed):
        """
        Generate a random causal graph
        Arguments:
            no_of_nodes: number of nodes in the causal graph
            no_of_edges: number of edges in the causal graph
            seed: seed for the random number generator
        Output:
            A CausalGraph object
        """
        dag = None
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
