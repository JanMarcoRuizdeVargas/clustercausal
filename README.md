# Cluster DAGs as background knowledge for causal discovery

This repository is being developed as part of my master thesis at TUM. 

I use the novel C-DAG (https://arxiv.org/abs/2202.12263) framework as background knowledge for causal discovery. 

Find my work in progress thesis at https://sharelatex.tum.de/read/npjkjggtqffh

Installation: 

Clone the repository, create a  virtual environment
```
python -m venv env
```

activate the virtual environment
```
env\Scripts\activate
```

and install the requirements
```
pip install -r requirements.txt
```

Usage: 

For  custom graphs see Cluster_PC_example.ipynb

For simulation studies and evaluation see Cluster_PC_simulation_gridsearch.ipynb and Cluster_PC_simulation_mass_simulation.ipynb

Tests:    

```
pytest --cov
```
