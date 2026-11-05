# Lab 5 - versioning & experiment tracking

This lab concerns versioning data and models, as well as experiment tracking.
Those are core MLOps concepts, fostering reproducibility, reliability, and 
auditability of ML processes.

**Learning plan**
1. Data versioning
   - Data Version Control (DVC)
   - configuring remote data storage
   - versioning datasets
2. Experiment tracking
   - MLflow 3 introduction, MLflow Tracking
   - autologging, custom logging
   - linking experiments with DVC data versions
   - hyperparameter optimization with Optuna, tabular deep learning with TabM
   - analyzing & comparing experiments
   - Model Registry, model aliases

**Necessary software**

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- (optional) [Docker and Docker Compose](https://docs.docker.com/engine/install/), for running
  the MLflow server in a container

Note that you should also activate `uv` project and install dependencies with `uv sync`.

**Lab**

There are separate instructions for DVC (part 1) and MLflow (part 2).
DVC uses Markdown instructions in [first lab instruction file](LAB_INSTRUCTION_1_DVC.md).
MLflow uses Jupyter Notebook in [second lab instruction file](LAB_INSTRUCTION_2_MLFLOW.ipynb),
also available as [Markdown](LAB_INSTRUCTION_2_MLFLOW.md).

There is no homework, only lab this time :)

**Data**

We will be using [Ames housing](https://www.openintro.org/book/statdata/?data=ames) dataset
about house prices in 2006-2010 in Ames, Iowa.
