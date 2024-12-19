import sys
import os

# Add the parent directory of the clustercausal package to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

import argparse
import yaml

# from clustercausal.experiments.ExperimentRunner import ExperimentRunner
from clustercausal.experiments.ExperimentRunnervsFCITiers import (
    ExperimentRunner,
)


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Running experiment for clustercausal"
    )

    # Add an argument for the config file path
    parser.add_argument(
        "config_file", type=str, help="Path to the config file in YAML format."
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the config from the specified file
    config = load_config(args.config_file)

    # Use the configuration as needed
    experiment = ExperimentRunner(args.config_file)
    experiment.run_gridsearch_experiment()


if __name__ == "__main__":
    main()
