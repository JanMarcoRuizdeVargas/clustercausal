import causallearn


class ExperimentRunner:
    """
    A class to run experiments in various configurations
    """

    def __init__(self):
        """
        Initialize the experiment runner
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
