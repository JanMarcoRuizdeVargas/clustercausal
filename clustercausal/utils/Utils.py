import numpy as np
import random
import os
import yaml
import pandas as pd
import pickle

import causallearn

from clustercausal.clusterdag.ClusterDAG import ClusterDAG


def draw_graph(nodes, edges):
    artificial_mapping = {}
    for node_name in nodes:
        artificial_mapping[node_name] = [node_name]
    cdag = ClusterDAG(artificial_mapping, edges)
    cdag.cluster_graph.draw_pydot_graph()


def generate_gaussian_anm(
    nodes, edges, num_samples=10000, seed=None, edge_weights=None
):
    n = len(nodes)
    node_map = {}
    for i in range(n):
        node_map[nodes[i]] = i
    data = np.zeros((num_samples, n))
    rng = np.random.default_rng(seed)
    if edge_weights is None:
        edge_weights = {}
        for edge in edges:
            edge_weights[edge] = rng.choice([-3, -2, -1, 1, 2, 3])
    for node in nodes:
        influence = np.zeros(num_samples)
        for edge in edges:
            if edge[1] == node:
                influence += edge_weights[edge] * data[:, node_map[edge[0]]]
        sample = rng.normal(size=num_samples)
        data[:, node_map[node]] = (
            influence + sample
        )  # np.random.normal(size = num_samples)
    return data, edge_weights


def is_valid_clustering(cdag, causal_graph):
    """
    Checks if a CDAG is a valid clustering of a causal graph

    Parameters
    ----------
    cdag : CDAG
        instance of CDAG class
    causal_graph : CausalGraph
        instance of CausalGraph class
    """
    pass


def load_experiment(experiment_folder):
    # Path to the results.yaml file in the first directory
    result_yaml = os.path.normpath(
        os.path.join(experiment_folder, "results.yaml")
    )

    # Check if results.yaml exists
    if os.path.exists(result_yaml):
        # Open the results.yaml file with YAML
        with open(result_yaml, "r") as file:
            result_dict = yaml.load(file, Loader=yaml.FullLoader)
        if result_dict == None:
            raise ValueError(
                "results.yaml is not available, experiment folder is empty"
            )
    return result_dict


def load_data(directory):
    # directory = gridsearch_directory
    # e.g. clustercausal/experiments/results/ClusterPC_2023-08-16 14-23-07.067298
    # Define the base directory
    columns = None
    experiment_folders = os.listdir(directory)
    for experiment in experiment_folders:
        experiment_path = os.path.join(directory, experiment)
        result_dict = load_experiment(experiment_path)
        if columns is None:
            columns = []
            for key in result_dict["base_evaluation_results"].keys():
                columns.append("base_" + key)
            for key in result_dict["cluster_evaluation_results"].keys():
                columns.append("cluster_" + key)
            for key in result_dict["settings"].keys():
                columns.append(key)
            data = pd.DataFrame(columns=columns)
        values = []
        for upper_key in result_dict:
            for key in result_dict[upper_key]:
                values.append(result_dict[upper_key][key])
        data = pd.concat(
            [data, pd.DataFrame([values], columns=columns)], ignore_index=True
        )
    for col in data.columns:
        if col not in [
            "dag_method",
            "distribution_type",
            "scm_method",
            "seed",
            "weight_range",
        ]:
            data[col] = data[col].astype(float)
    return data


def load_experiment_graphs(experiment_folder):
    """
    Loads the base_est, cluster_est and cluster_dag from an experiment folder
    Parameters
    ----------
    experiment_folder : str
        Path to the experiment folder

    Returns
    -------
    base_est : CausalGraph
        Estimated causal graph from base method
    cluster_est : CausalGraph
        Estimated causal graph from cluster method
    cluster_dag : ClusterDAG
        Estimated cluster DAG from cluster method
    """
    base_est_path = os.path.join(experiment_folder, "base_est_graph.pkl")
    cluster_est_path = os.path.join(experiment_folder, "cluster_est_graph.pkl")
    cluster_dag_path = os.path.join(experiment_folder, "cluster_dag.pkl")
    if os.path.exists(base_est_path):
        with open(base_est_path, "rb") as file:
            base_est_graph = pickle.load(file)
    if os.path.exists(cluster_est_path):
        with open(cluster_est_path, "rb") as file:
            cluster_est_graph = pickle.load(file)
    if os.path.exists(cluster_dag_path):
        with open(cluster_dag_path, "rb") as file:
            cluster_dag = pickle.load(file)
    return base_est_graph, cluster_est_graph, cluster_dag


def causallearn_to_nx_adjmat(adjmat: np.ndarray) -> np.ndarray:
    """
    Convert a causallearn adjacency matrix to a networkx adjacency matrix
    i.e. tail ends get converted from -1 to 0
    Parameters
    ----------
    adjmat : np.ndarray
        Adjacency matrix from causallearn

    Returns
    -------
    nx_adjmat : np.ndarray
        Adjacency matrix for networkx
    """
    nx_adjmat = np.zeros(adjmat.shape)
    for i in range(adjmat.shape[0]):
        for j in range(adjmat.shape[1]):
            if adjmat[i, j] == -1 and adjmat[j, i] == 1:
                nx_adjmat[i, j] = 1
                nx_adjmat[j, i] = 0
            if adjmat[i, j] == -1 and adjmat[j, i] == -1:
                nx_adjmat[i, j] = 1
                nx_adjmat[j, i] = 1
    return nx_adjmat


def nx_to_causallearn_adjmat(adjmat: np.ndarray) -> np.ndarray:
    """
    Convert a networkx adjacency matrix to a causallearn adjacency matrix
    i.e. tail ends get converted from 0 to -1
    Parameters
    ----------
    adjmat : np.ndarray
        Adjacency matrix from networkx

    Returns
    -------
    causallearn_adjmat : np.ndarray
        Adjacency matrix for causallearn
    """
    causallearn_adjmat = np.zeros(adjmat.shape)
    for i in range(adjmat.shape[0]):
        for j in range(adjmat.shape[1]):
            if adjmat[i, j] == 1 and adjmat[j, i] == 0:
                causallearn_adjmat[i, j] = -1
                causallearn_adjmat[j, i] = 1
            if adjmat[i, j] == 1 and adjmat[j, i] == 1:
                causallearn_adjmat[i, j] = -1
                causallearn_adjmat[j, i] = -1
    return causallearn_adjmat
