# Lab 6 - data & models testing

This lab concerns testing data quality, model behavioral testing, and explainability.
Those operations are crucial part of data-centric AI approach, and constitute a large
part of MLOps quality assurance.

**Learning plan**
1. Data quality testing:
   - smoke tests and manual data cleaning
   - data contracts, Pandera
   - confident learning, detecting & fixing label issues, near duplicates and outliers
   - CleanLab
2. Model testing:
   - decision models as zero-shot classifiers, basal
   - classification metrics, calibration, confidence thresholds
   - behavioral testing with input perturbations
   - LangTest
3. ML explainability:
   - local explainability
   - gradient-based and perturbation-based attribution methods
   - Captum

**Necessary software**
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

Note that you should also activate `uv` project and install dependencies with `uv sync`.
The decision model (1.6B parameters) runs on an NVIDIA GPU, on Apple Silicon, or on CPU with at least
16 GB of RAM.

**Lab**

See [lab instruction](LAB_INSTRUCTION.ipynb), also available as [Markdown](LAB_INSTRUCTION.md).

**Homework**

See [homework instruction](HOMEWORK.md).

**Data**

In the lab, we will be using [booking.com reviews dataset](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe?resource=download).
In the homework, you will use [Banking77 dataset](https://huggingface.co/datasets/PolyAI/banking77).