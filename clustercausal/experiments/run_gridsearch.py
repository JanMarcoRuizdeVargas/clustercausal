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
    config_path = "clustercausal\experiments\configs\debug_config.yaml"
    # config_path = "clustercausal\experiments\configs\cluster_pc_1.yaml"
    config_path = config_path.replace("\\", "/")
    experiment = ExperimentRunner(config_path)
    experiment.run_gridsearch_experiment()
