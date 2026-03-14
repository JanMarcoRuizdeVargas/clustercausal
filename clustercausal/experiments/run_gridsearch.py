import itertools
import yaml
import pickle
import pandas as pd
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)

# from clustercausal.experiments.ExperimentRunner import ExperimentRunner
from clustercausal.experiments.ExperimentRunnervsFCITiers import (
    ExperimentRunner,
)
from clustercausal.utils.Utils import *

if __name__ == "__main__":
    n_jobs = 20
    print("STARTING 1st EXPERIMENT")
    config_path = (
        "clustercausal\experiments\configs\cluster_pc_simulation_1new_kpc.yaml"
    )
    config_path = config_path.replace("\\", "/")
    experiment = ExperimentRunner(config_path)
    experiment.run_gridsearch_experiment(n_jobs=n_jobs)

    # print("STARTING 2nd EXPERIMENT")
    # config_path = (
    #     "clustercausal\experiments\configs\cluster_pc_simulation_2new.yaml"
    # )
    # config_path = config_path.replace("\\", "/")
    # experiment = ExperimentRunner(config_path)
    # experiment.run_gridsearch_experiment(n_jobs=n_jobs)

    # print("STARTING 3rd EXPERIMENT")
    # config_path = "clustercausal\experiments\configs\cluster_vsfcitiers_simulation_3new.yaml"
    # config_path = config_path.replace("\\", "/")
    # experiment = ExperimentRunner(config_path)
    # experiment.run_gridsearch_experiment(n_jobs=n_jobs)

    # print("STARTING 4th EXPERIMENT")
    # config_path = "clustercausal\experiments\configs\cluster_vsfcitiers_on_tiers_simulation_4new.yaml"
    # config_path = config_path.replace("\\", "/")
    # experiment = ExperimentRunner(config_path)
    # experiment.run_gridsearch_experiment(n_jobs=n_jobs)

    # print("STARTING 5th EXPERIMENT")
    # config_path = "clustercausal\experiments\configs\cluster_pc_5.yaml"
    # config_path = config_path.replace("\\", "/")
    # experiment = ExperimentRunner(config_path)
    # experiment.run_gridsearch_experiment()
