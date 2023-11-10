import itertools
import yaml
import os
import pickle
import pandas as pd
from clustercausal.utils.Utils import load_data
from clustercausal.experiments.ExperimentRunner import ExperimentRunner


def load_experiment_folder(directory):
    folder_paths = []
    data = None
    for filename in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, filename)):
            folder_paths.append(os.path.join(directory, filename))
    for folder in folder_paths:
        if data is None:
            data = load_data(folder)
        else:
            data = pd.concat([data, load_data(folder)])
    return data


def load_experiment_graphs(directory):
    """
    Loads base_est_graph, cluster_est_graph and cluster_dag from
    an experiment folder
    """
    with open(directory, "rb") as file:
        base_est_graph = pickle.load(
            os.path.join(directory, "base_est_graph.pkl")
        )
        cluster_est_graph = pickle.load(
            os.path.join(directory, "cluster_est_graph.pkl")
        )
        cluster_dag = pickle.load(os.path.join(directory, "cluster_dag.pkl"))
    return base_est_graph, cluster_est_graph, cluster_dag
