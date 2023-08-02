import causallearn
import yaml
import itertools
import os
import datetime

from causallearn.search.ConstraintBased.PC import pc

from clustercausal.experiments.Simulator import Simulator
from clustercausal.experiments.Evaluator import Evaluator
from clustercausal.algorithms.ClusterPC import ClusterPC


class ExperimentRunner:
    """
    A class to run experiments in various configurations
    """

    def __init__(self, config_path, algorithm="ClusterPC"):
        """
        Args:
            config_path (str): path to the experiment configuration file


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
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.discovery_alg = self.config["discovery_alg"]
        self.config.pop("discovery_alg")

        if "linear" in self.config["scm_method"]:
            self.linear_config = self.config.copy()
            self.linear_config["scm_method"] = ["linear"]
            self.linear_config.pop("lin_distribution_type")
            self.linear_config.pop("nonlin_distribution_type")
            self.linear_config["distribution_type"] = self.config[
                "lin_distribution_type"
            ]
        if "nonlinear" in self.config["scm_method"]:
            self.nonlinear_config = self.config.copy()
            self.nonlinear_config["scm_method"] = ["nonlinear"]
            self.nonlinear_config.pop("lin_distribution_type")
            self.nonlinear_config.pop("nonlin_distribution_type")
            self.nonlinear_config["distribution_type"] = self.config[
                "nonlin_distribution_type"
            ]

        num_lin_experiments = 1
        for key in self.linear_config.keys():
            num_lin_experiments *= len(self.linear_config[key])
        num_nonlin_experiments = 1
        for key in self.nonlinear_config.keys():
            num_nonlin_experiments *= len(self.nonlinear_config[key])
        num_experiments = num_lin_experiments + num_nonlin_experiments
        print(f"Number of experiments: {num_experiments}")

    def run_gridsearch_experiment(self):
        """
        Run experiments with a grid of configurations
        """
        if self.linear_config is not None:
            lin_param_configuration = list(
                itertools.product(*self.linear_config.values())
            )
            for params in lin_param_configuration:
                self.run_experiment(params)

        if self.nonlinear_config is not None:
            nonlin_param_configuration = list(
                itertools.product(*self.nonlinear_config.values())
            )
            for params in nonlin_param_configuration:
                self.run_experiment(params)

    def run_experiment(self, params):
        """
        Run an experiment
        # TODO add different independence tests
        """
        param_names = list(
            self.linear_config.keys()
        )  # for names doesn't matter linear or nonlinear
        param_dict = dict(zip(param_names, params))
        print(f"Running experiment with parameters: {param_dict}")
        # run simulation
        simulation = Simulator(**param_dict)
        cluster_dag = simulation.run()
        # run causal discovery
        cluster_pc = ClusterPC(
            cdag=cluster_dag,
            data=cluster_dag.data,
            alpha=0.05,
            indep_test="fisherz",
            verbose=False,
            show_progress=False,
        )
        cluster_est_graph = cluster_pc.run()
        base_est_graph = pc(
            cluster_dag.data, alpha=0.05, verbose=False, show_progress=False
        )
        # evaluate causal discovery
        cluster_evaluation = Evaluator(
            truth=cluster_dag.true_dag.G, est=cluster_est_graph.G
        )
        (
            cluster_adjacency_confusion,
            cluster_arrow_confusion,
            cluster_shd,
        ) = cluster_evaluation.get_causallearn_metrics()
        cluster_adjacency_confusion = {
            f"adj_{k}": v for k, v in cluster_adjacency_confusion.items()
        }
        cluster_arrow_confusion = {
            f"arrow_{k}": v for k, v in cluster_arrow_confusion.items()
        }
        cluster_evaluation_results = {
            **cluster_adjacency_confusion,
            **cluster_arrow_confusion,
            "cluster_shd": cluster_shd,
        }

        base_evaluation = Evaluator(
            truth=cluster_dag.true_dag.G, est=base_est_graph.G
        )
        (
            base_adjacency_confusion,
            base_arrow_confusion,
            base_shd,
        ) = base_evaluation.get_causallearn_metrics()
        base_adjacency_confusion = {
            f"adj_{k}": v for k, v in base_adjacency_confusion.items()
        }
        base_arrow_confusion = {
            f"arrow_{k}": v for k, v in base_arrow_confusion.items()
        }
        base_evaluation_results = {
            **base_adjacency_confusion,
            **base_arrow_confusion,
            "base_shd": base_shd,
        }

        # save results
        gridsearch_name = f"{self.discovery_alg[0]}_{str(datetime.datetime.now()).replace(':', '-')}"
        folder_name = (
            param_dict["dag_method"]
            + f" {param_dict['n_nodes']} nodes"
            + f" {param_dict['distribution_type']} distribution"
        )
        file_path = os.path.join(
            "clustercausal",
            "experiments",
            "results",
            gridsearch_name,
            folder_name,
        )
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        file_name_cluster = "cluster_evaluation_results.yaml"
        file_path_cluster = os.path.join(file_path, file_name_cluster)
        with open(file_path_cluster, "w") as file:
            yaml.dump(cluster_evaluation_results, file)

        file_name_base = "base_evaluation_results.yaml"
        file_path_base = os.path.join(file_path, file_name_base)
        with open(file_path_base, "w") as file:
            yaml.dump(base_evaluation_results, file)
