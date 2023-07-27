import causallearn


class ExperimentRunner:
    """
    A class to run experiments in various configurations
    """

    def __init__(self, dag_methods):
        """
        Initialize the experiment runner configuration
        dag_methods: ['erdos_renyi', 'scale_free', 'bipartite', 'hierarchical']
        n_nodes: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        n_clusters: [2, 3, 5, 10, 20, 30, 50] # high clust counts get ommitted for low node counts
        edges_added_on_nodes: [-2, 5, 10, 30, 50, 100]
        weight_ranges = [(0.5,2), (-1,1), (-1,2)]
        distribution_types_linear: ['gauss', 'exp', 'gumbel', 'uniform', 'logistic']
        distribution_types_nonlinear: ['lmp', 'mim', 'gp', 'gp-add', 'quadratic']
        scm_types: ['linear', 'nonlinear']
        noise_scale: [0.3, 1, 2, 5]
        """
        pass

    def run_gridsearch_experiment(self, configurations):
        """
        Run experiments with a grid of configurations
        """
        for config in configurations:
            self.run_experiment(config)
        pass

    def run_experiment(self):
        """
        Run an experiment
        """

        # run simulation
        # run causal discovery
        # return cpdag or pag
