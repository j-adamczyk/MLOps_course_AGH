# Lab 1 - MLOps introduction

This lab introduces you to basic MLOps tools and local model serving. We will
go over modern Python and DevOps tools and their usage in MLOps. This will cover
setting up an entire application from scratch to get a fully functional and
containerized model running locally.

**Learning plan**
1. Python development tools
   - dependency management, `uv`
   - code versioning, `git` and GitHub
   - pre-commit hooks
   - testing, `pytest`
   - serving models with FastAPI
2. DevOps tools
   - environment variables management, `.env` files
   - secrets management, `sops`
   - containers, Docker, Docker Compose

**Necessary software**
- [Docker and Docker Compose](https://docs.docker.com/engine/install/), 
  also [see those post-installation notes](https://docs.docker.com/engine/install/linux-postinstall/)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [sops](https://github.com/getsops/sops)

Note that you should also activate `uv` project and install dependencies with `uv sync`.

**Lab**

See [lab instruction](LAB_INSTRUCTION.md). Laboratory is worth 5 points.

**Homework**

See [homework instruction](HOMEWORK.md). Homework is worth 10 points.
