import numpy as np
import networkx as nx

from castle.datasets.simulator import IIDSimulation, DAG

from clustercausal.experiments.Simulator import Simulator


def test_simulator():
    # list_of_dag_nodes = [5, 10, 15, 20]
    # list_of_dag_edges = [3, 5, 7, 10]
    list_of_dag_nodes = [5]
    list_of_dag_edges = [5]
    list_of_dag_methods = [
        "erdos_renyi",
        "scale_free",
        "bipartite",
        "hierarchical",
        # "low_rank"
    ]
    list_of_linear_sem_types = [
        "gauss",
        "exp",
        "gumbel",
        "uniform",
        "logistic",
    ]
    list_of_nonlinear_sem_types = ["mlp", "mim", "gp", "gp-add", "quadratic"]
    list_of_scm_methods = ["linear", "nonlinear"]
    list_of_noise_scales = [0.1, 0.5, 1.0, 2.0]
    for n_nodes in list_of_dag_nodes:
        for n_edges in list_of_dag_edges:
            for dag_method in list_of_dag_methods:
                for scm_method in list_of_scm_methods:
                    for noise_scale in list_of_noise_scales:
                        if scm_method == "linear":
                            for sem_type in list_of_linear_sem_types:
                                simulation = Simulator()
                                true_dag, data = simulation.run(
                                    true_dag=None,
                                    n_nodes=n_nodes,
                                    n_edges=n_edges,
                                    dag_method=dag_method,
                                    weight_range=(-1, 2),
                                    distribution_type=sem_type,
                                    scm_method=scm_method,
                                    sample_size=100,
                                    seed=42,
                                    noise_scale=noise_scale,
                                )
                                nx_graph = nx.from_numpy_array(
                                    true_dag.adjacency_matrix,
                                    create_using=nx.DiGraph,
                                )
                                assert data.shape == (100, n_nodes)
                                assert (
                                    nx.is_directed_acyclic_graph(nx_graph)
                                    == True
                                )
                        if scm_method == "nonlinear":
                            for sem_type in list_of_nonlinear_sem_types:
                                simulation = Simulator()
                                true_dag, data = simulation.run(
                                    true_dag=None,
                                    n_nodes=n_nodes,
                                    n_edges=n_edges,
                                    dag_method=dag_method,
                                    weight_range=(-1, 2),
                                    distribution_type=sem_type,
                                    scm_method=scm_method,
                                    sample_size=100,
                                    seed=42,
                                    noise_scale=noise_scale,
                                )
                                nx_graph = nx.from_numpy_array(
                                    true_dag.adjacency_matrix,
                                    create_using=nx.DiGraph,
                                )
                                assert data.shape == (100, n_nodes)
                                assert (
                                    nx.is_directed_acyclic_graph(nx_graph)
                                    == True
                                )


def test_simulator_generate_dag():
    # TODO
    pass


def test_simulator_generate_data():
    # TODO
    pass
