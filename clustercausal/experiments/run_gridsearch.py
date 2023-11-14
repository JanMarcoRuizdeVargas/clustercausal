import itertools
import yaml
import pickle
import pandas as pd
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

from clustercausal.experiments.ExperimentRunner import ExperimentRunner
from clustercausal.utils.Utils import *

if __name__ == "__main__":
    # config_path = "clustercausal\experiments\configs\debug_config.yaml"
    # print("STARTING 1st EXPERIMENT")
    config_path = (
        "clustercausal\experiments\configs\cluster_pc_mass_simulation.yaml"
    )
    config_path = config_path.replace("\\", "/")
    experiment = ExperimentRunner(config_path)
    experiment.run_gridsearch_experiment()

    # print("STARTING 2nd EXPERIMENT")
    # config_path = "clustercausal\experiments\configs\cluster_pc_2.yaml"
    # config_path = config_path.replace("\\", "/")
    # experiment = ExperimentRunner(config_path)
    # experiment.run_gridsearch_experiment()

    # print("STARTING 3rd EXPERIMENT")
    # config_path = "clustercausal\experiments\configs\cluster_pc_3.yaml"
    # config_path = config_path.replace("\\", "/")
    # experiment = ExperimentRunner(config_path)
    # experiment.run_gridsearch_experiment()

    # print("STARTING 4th EXPERIMENT")
    # config_path = "clustercausal\experiments\configs\cluster_pc_4.yaml"
    # config_path = config_path.replace("\\", "/")
    # experiment = ExperimentRunner(config_path)
    # experiment.run_gridsearch_experiment()

    # print("STARTING 5th EXPERIMENT")
    # config_path = "clustercausal\experiments\configs\cluster_pc_5.yaml"
    # config_path = config_path.replace("\\", "/")
    # experiment = ExperimentRunner(config_path)
    # experiment.run_gridsearch_experiment()
