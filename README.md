# Cluster DAGs as background knowledge for causal discovery

This repository is being developed as part of my master thesis at TUM. 

I use the novel C-DAG (https://arxiv.org/abs/2202.12263) framework as background knowledge for causal discovery. 

Find my work in progress thesis at https://sharelatex.tum.de/read/npjkjggtqffh

Installation (requires Python 3.10): 

Clone the repository in a folder of your choice:
```
git clone https://github.com/JanMarcoRuizdeVargas/clustercausal.git
```

Change directory to the repository:
```
cd .\clustercausal\
```

Create a  virtual environment:
```
python -m venv env
```

Activate the virtual environment:
```
.\env\Scripts\activate
```

Install the requirements:
```
pip install -r requirements.txt
```

To be able to calculate SID in the metrics one needs to install R (version 4.3.1 is recommended). 

Usage: 

For  custom graphs see Cluster_PC_example.ipynb. 

For simulation studies and evaluation see Cluster_PC_simulation_gridsearch.ipynb and Cluster_PC_simulation_mass_simulation.ipynb. 

Tests:    

```
pytest --cov
```
