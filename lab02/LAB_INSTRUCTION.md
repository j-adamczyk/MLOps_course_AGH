# Laboratory 2 instruction

## PostgreSQL setup (1 point)

Postgres is a relational database, with a typical client-server architecture. We will
use Docker and Docker Compose to set it up and manage it. Using Docker has several
advantages, but primarily:
- you can simulate many databases at the same time
- easy reset and cleanup
We assume you have Docker and Docker Compose installed, see README for this lab.

### Docker Compose file

From start, we will use Docker Compose. It allows easy management of many Docker
containers, e.g. you can run database, backend server, and frontend. As it uses
convenient YAML file for configuration, it's useful even for single containers.
See `compose.yaml` for an example. Note that YAML files can have either `.yml` or
`.yaml` extension.

Default name for configuration file is `compose.yaml`. If you use it, you don't have
to pass this name to further commands. Otherwise, you'll need `--file` option.

Let's see an example:
```yaml
services:
  postgres:
    image: postgres:17
    restart: always
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=postgres
    ports:
      - '5432:5432'
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

`services` is the list of services, run in separate Docker containers. Here,
we have just one service named `postgres`, but you could have more. For this service,
we provide its configuration:
1. `image` - Docker image name. Here, we use Postgres version 17. See [Postgres DockerHub](https://hub.docker.com/_/postgres) for more.
2. `restart` - behavior in case of problems. Setting it to `always` will immediately restart
   the service (and underlying Docker container) if necessary.
3. `environment` - environment variables, the most typical way of passing configuration to services.
4. `ports` - maps host machine port to container port. By default, containers are run in isolation,
   and you would treat them as a completely separate network. Mapping ports sets up connectivity
   on those selected ports. This needs to be done explicitly, even for identical ports like we do
   here. Syntax is: `host_port:container_port`.
5. `volumes` - volumes are persistent data stores for Docker containers. They keep the data
   even if container is restarted or killed. This way, you have an actual database, saved on disk.
   They are mapped like ports. Here, we note that volume `pgdata` corresponds to path `/var/lib/postgresql/data`
   inside the container, where Postgres saves its data.

`volumes` defines the list of volumes managed by Docker Compose. If you want to use a volume for
a service, it also needs to be listed here. By default, Docker keeps volume locally on the host
machine (you could provide `driver: local` as configuration inside `pgdata`), but it can also be
e.g. remote cloud storage. This way, you can use Docker Compose for production deployments.

### Docker Compose cheatsheet

Get help: `docker compose --help`

Start: `docker compose up`\
Start detached (in the background): `docker compose up --detach`\
Restart: `docker compose restart`\
Kill: `docker compose down`

List containers: `docker compose ps`\
Print container logs: `docker compose logs SERVICE_NAME`\
Continuously follow logs: `docker compose logs SERVICE_NAME --follow`

Remove stopped containers: `docker container prune`\
Remove unused container images: `docker image prune`\
Remove unused volumes: `docker volume prune`\
Remove everything, total cleanup: `docker system prune --all --volumes`

Remember that ctrl+C interrupts current program in the shell, ctrl+Z suspends it and
sends to the background, and q exits interactive listings.

For bigger projects, particularly with non-default config YAML name, making aliases or
a Makefile for those is often useful.

### psql CLI

Postgres client uses `psql` CLI tool. Its most important options are related to database connection:
- `-h HOSTNAME` / `--host=HOSTNAME` - database host
- `-p PORT` / `--port=PORT` - database port, default 5432
- `-U USERNAME` / `--username=USERNAME` - database user name
- `-d DBNAME` / `--dbname=DBNAME` - database name

By default, Postgres looks for local database (run by host), on port 5432, with the same username as
current user, `postgres` database. It will prompt for database password.

Postgres can have many separate logical databases on one database server. This way, you can e.g.
separate clients, but most people prefer to have just one database. Note that this is separate thing
from schema, which are like directories for tables in Postgres.

After you connect, you can directly run SQL queries. You can also run administrative commands or
special commands, typically starting with backslash `\`. In Postgres, individual queries are ended
or separated with `;`.

Alternatively, you can run queries from string: `psql -c 'SELECT * FROM users;'`. String query can
be either in apostrophes or quotes (`'`, `"`), depending on a use case. Remember that column names
must be in quotes and strings in apostrophes, e.g. `WHERE "Stop:Station code" = 'RTD'`. If necessary,
you can escape a quote by repeating it, i.e. `''value''` -> 'value', `""value""` -> "value".

If you want to run query from .sql file, use `--file` argument and pass relative path to the file.

### psql cheatsheet

* Connect interactively: `psql -h localhost -U postgres`
* Run query string: `psql -h localhost -U postgres -c 'query'`
* Run query from file: `psql -h localhost -U postgres -f 'file_path'`

* List tables in database (`\dt'`): `psql -h localhost -U postgres -c '\dt'`
* Get table schema: (`\d table_name'`): `psql -h localhost -U postgres -c '\d table_name'`
* Get number of rows: `psql -h localhost -U postgres -c 'SELECT COUNT(*) FROM table_name'`

This cheatsheet assumes that you use host and username as defined above. 

### Running Postgres with Docker Compose and connecting

Run `docker compose up --detach` to start the container. Check with `docker compose ps` that it runs.

To connect, run `psql --host=localhost --username=postgres`, and then `SELECT * FROM services;`.
Since database is empty, it will error. To exit, run `\q`.

Query in `data/services_table.sql` will create a table for Dutch train services, along with
some indexes useful for faster searching later. It does not have a primary key, which may be
a bit weird, but this makes sense when you look at the [data definition](https://www.rijdendetreinen.nl/en/open-data/train-archive).
Also, in Postgres primary key is just index with UNIQUE constraint, so we don't lose much.

Run the query with:
```
psql --host localhost --username postgres --file data/services_table.sql
```

## 1. PostgreSQL - MLOps usage (1 point)

### Bulk import/export

Postgres, as OLTP database, supports efficient read and write operations on individual rows.
However, **bulk operations** are typically much more efficient when done on whole batches
of data. This is quite common in MLOps and big data engineering, e.g.:
- dump whole table as CSV
- save query results on disk
- insert batch of model predictions into table
- load embeddings for text corpus

Inserting or reading individual rows would be very inefficient, as it uses a lot of tiny
transactions, which are simply unnecessary here. Postgres offers optimized COPY command
for batch operations, which supports CSV as input or output.

[Rijden de Treinen](https://www.rijdendetreinen.nl/en/open-data) open data initiative
contains comprehensive data about the Dutch railway network. Among other things, it tracks
all train services: station information, departure and arrival times, delays etc. Download
and unpack the data about 2024 train services by running:
```
wget https://opendata.rijdendetreinen.nl/public/services/services-2024.csv.gz -O data/services-2024.csv.gz && \
gzip -f -d data/services-2024.csv.gz
```
You can take a look at it e.g. in LibreOffice Calc, but it contains a little under 22 million
rows, so you won't be able to load it all. Postgres will easily handle this amount of data.

COPY command has a following syntax:
`\copy table_name FROM 'csv_path.csv' DELIMITER ',' CSV HEADER;`
It uses client-side file, so you provide your own file. `COPY` command would use server-side
file, which is much rarer. Of course, you can have different delimiters or no header, and you
can specify column names in parentheses after `table_name` if necessary.

To copy it into the table, run:
```
psql -h localhost -U postgres -c "\copy services FROM 'data/services-2024.csv' DELIMITER ',' CSV HEADER"
```

To export a table in CSV format, we also use COPY, and we can also save any query results this way:
```
\copy (query) TO file_path WITH CSV DELIMITER ',' HEADER
```

### psycopg library

[psycopg3](https://www.psycopg.org/psycopg3/docs/_), also known as just `psycopg`, is a standard library
for connecting to Postgres from Python. It is also used by other frameworks, e.g. SQLAlchemy and peewee.

Go to [notebook_01_psycopg.ipynb](notebook_01_psycopg.ipynb).

## 2. DuckDB and file formats (3 points)

You already have DuckDB, Pandas, and pyarrow installed and ready to use. DuckDB, as an
embedded, in-process database, does not require any additional setup, it's just run from
Python code.

We will use additional data about 2023 train disruptions. To download it, run:
```commandline
wget https://opendata.rijdendetreinen.nl/public/disruptions/disruptions-2023.csv -O data/disruptions-2023.csv
```

Go to [notebook_02_duckdb.ipynb](notebook_02_duckdb.ipynb).
