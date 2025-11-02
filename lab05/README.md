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
   - MLflow introduction, MLflow Tracking
   - autologging, custom logging
   - analyzing & comparing experiments

**Necessary software**

- [uv](https://docs.astral.sh/uv/getting-started/installation/)

Note that you should also activate `uv` project and install dependencies with `uv sync`.

**Lab**

There are separate instructions for DVC (part 1) and MLFlow (parts 2-3).
DVC uses Markdown instructions in [first lab instruction file](LAB_INSTRUCTION_1_DVC.md).
MLflow uses Jupyter Notebook in [second lab instruction file](LAB_INSTRUCTION_2_MLFLOW.ipynb).

There is no homework, only lab this time :)

**Data**

We will be using [Ames housing](https://www.openintro.org/book/statdata/?data=ames) dataset
about house prices in 2006-2010 in Ames, Iowa.
