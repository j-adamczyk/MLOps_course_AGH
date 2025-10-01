# Lab 2 - Databases & file formats

This lab concerns working with databases and files, which are data sources and
processing engines for MLOps. Analytics will be the main focus, as an essential
part of data preparation and exploratory data analysis (EDA).

**Learning plan**
1. PostgreSQL
   - example of server-based OLTP database
   - setting up in Docker
   - connecting with CLI and Python (`psycopg` library)
   - loading, querying, and saving data
2. DuckDB
   - example of in-process OLAP database
   - setting up in Python (`duckdb` library)
   - querying files and databases
   - interoperability with Pandas
3. File formats
   - CSV, JSON, JSON Lines, Parquet
   - reading, writing
   - size & speed comparison

**Necessary software**
- [Docker and Docker Compose](https://docs.docker.com/engine/install/), 
  also [see those post-installation notes](https://docs.docker.com/engine/install/linux-postinstall/)
- Postgres client, e.g. `sudo apt install postgresql-client`
  ([more details](https://askubuntu.com/questions/1040765/how-to-install-psql-without-postgres))
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

Note that you should also activate `uv` project and install dependencies with `uv sync`.

**Lab**

See [lab instruction](LAB_INSTRUCTION.md). Laboratory is worth 5 points.

**Homework**

See [homework instruction](HOMEWORK.md). Homework is worth 10 points.

**Data**

We will be using [Rijden de Treinen](https://www.rijdendetreinen.nl/en/open-data) data
about the Dutch railway network.
