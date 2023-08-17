import numpy as np
import random
import os
import yaml
import pandas as pd

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
    return data
