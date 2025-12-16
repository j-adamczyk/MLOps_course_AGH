## Lab 11 - Deployment & CI/CD

This lab focuses on deployment of ML models and building continuous integration (CI)
and continuous deployment (CD) pipelines for that purpose. Those operations allow safe,
automated and efficient updates of models to production environment.

We will use GitHub Actions for implementing CI/CD pipelines due to their simplicity and
popularity. Prior to deployment itself, we will use ONNX tools to compile and optimize
our model.

**Learning Plan**
1. GitHub Actions workflows
2. CI/CD pipelines for Python applications
3. ONNX format and runtime
4. Lightweight Docker images for ML inference

**Necessary software**
- AWS account
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
- GitHub Account
- [Docker and Docker Compose](https://docs.docker.com/engine/install/),
  also [see those post-installation notes](https://docs.docker.com/engine/install/linux-postinstall/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

Note that you should also create and activate `uv` project.

**Lab**

See [lab instruction](LAB_INSTRUCTION.md).

**Homework**

See [homework instruction](HOMEWORK.md).
