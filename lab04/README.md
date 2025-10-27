# Lab 4 - Vector databases

This lab concerns using vector search and vector databases. Those are frequently
used tools for searching, recommendation, and retrieval-augmented generation (RAG).
We will go over various approaches to creating vector search indexes and working
with them.

**Learning plan**
1. Vector search index
   - vector search as relational database index
   - Postgres, TimescaleDB, pgvectorscale
   - SQLAlchemy
   - text embeddings
   - vector search queries
2. Vector databases and RAG
   - Milvus
   - text chunking
   - external API integration
   - RAG

**Necessary software**
- [Docker and Docker Compose](https://docs.docker.com/engine/install/), 
  also [see those post-installation notes](https://docs.docker.com/engine/install/linux-postinstall/)
- Postgres client, e.g. `sudo apt install postgresql-client`
  ([more details](https://askubuntu.com/questions/1040765/how-to-install-psql-without-postgres))
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.11

Note that you should also activate `uv` project and install dependencies with `uv sync`.

**Lab**

See [lab instruction](LAB_INSTRUCTION.md).

**Homework**

See [homework instruction](HOMEWORK.md).

**Data**

We will be using [Steam Games Dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset)
about games published on Steam, as well as
[Amazon Berkeley Objects (ABO) Dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)
with data about objects available in the Amazon store.
