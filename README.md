# Cluster DAGs as background knowledge for causal discovery

This repository was developed as part of my master thesis at TUM. 

I use the novel C-DAG (https://arxiv.org/abs/2202.12263) framework as background knowledge for causal discovery. 

Find my thesis at https://sharelatex.tum.de/read/npjkjggtqffh

Installation (requires Python 3.10): 

Clone the repository in a folder of your choice:
```
git clone https://github.com/JanMarcoRuizdeVargas/clustercausal.git
```

Certain foldernames in the notebooks or in ```clustercausal/experiments/run_gridsearch.py``` might have to be changed to adjust from Windows to macOS or Linux to work properly.

 
Change directory to the repository(Windows):
```
cd .\clustercausal\
```
macOS:
```
cd clustercausal
```


Create a  virtual environment(Windows):
```
python -m venv env
```
macOS:
```
python3 -m venv env
```


Activate the virtual environment(Windows):
```
.\env\Scripts\activate
```
macOS:
```
source env/bin/activate
```

(To deactivate the virtual environment after one is done, run ```deactivate``` for both Windows and macOS)


Install the requirements (same for Windows and macOS):
```
pip install -r requirements.txt
```

To be able to calculate SID in the metrics one needs to install R (version 4.3.1 is recommended). In addition, one needs to change ```os.environ[
    "R_HOME"
] = "C:\Program Files\R\R-4.3.1"``` in ```clustercausal/experiments/Evaluator.py``` to the path where R is installed.


Usage: 

For  custom graphs see Cluster_PC_example.ipynb. 

For simulation studies and evaluation see Cluster_PC_simulation_gridsearch.ipynb and Cluster_PC_simulation_mass_simulation.ipynb. 

Tests:    

```
pytest --cov
```
