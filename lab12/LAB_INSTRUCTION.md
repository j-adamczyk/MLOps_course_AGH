# Laboratory 12 instruction

## Introduction

This lab will cover building pipelines and workflows for ML, a core element of MLOps.
We will focus on [Apache Airflow](https://airflow.apache.org/), arguably the most popular
open source workflow orchestrator.

## Local Airflow setup

Airflow, as a framework orchestrating the work of other operations and tools, itself runs
a few processes. In local deployment mode:

- the scheduler process is just a regular Python process
- the backend database is SQLite, just a single file
- workers run as scheduler subprocesses

This aims to simplify local development and testing, but is not enough for a production
environment. We will do that later.

Create virtual environment via `uv sync` and activate it. Airflow is already added to the
dependencies list. It is highly modular, so that the basic package is lightweight, relying
on optional plugin packages.

By default, Airflow uses the `~/airflow` directory to host its local files, and is configured
through `airflow.cfg` file put there (or environment variables on launch). If it does not exist,
it will be generated on the first run.

Let's run it:

```bash
export AIRFLOW_HOME=`pwd`/airflow && airflow standalone
```

Airflow UI is available at http://localhost:8080/. However, it requires a password, which is
automatically generated on the first launch in `simple_auth_manager_passwords.json.generated`
file, next to `airflow/airflow.cfg`. A few key elements of that file are:

- `dags_folder` - where Airflow reads DAG (pipeline) definitions from
- `executor` - name of executors used, first one is the default
- `load_examples` - whether to load some example DAGs, `False` for production environments
- `fernet_key` - encryption key for keeping secrets in the Airflow database,
  using [Fernet algorithm](https://airflow.apache.org/docs/apache-airflow/stable/security/secrets/fernet.html)
- `xcom_backend` - backend used by XCom to pass data between tasks, by default local file system
- `sql_alchemy_conn` - connection string to the database
- `port` - port for UI server, by default 8080

Access the UI using the credentials from the file. Browse the example DAGs in the `Dags` tab, switch
to list view for more compact listing. Find `example_python_operator` DAG and trigger (run) it manually,
(upper right corner button). After execution, switch to the graph view in the upper left corner. Code
for this DAG is
available [on Airflow GitHub repository](https://github.com/trbs/airflow-examples/blob/master/dags/example_python_operator.py).

![01_dags_list.png](images/01_dags_list.png)

![02_example_dag.png](images/02_example_dag.png)

![03_dag_run.png](images/03_dag_run.png)

Airflow metadata database is SQLite in this case, stored in `airflow/airflow.db` file. It is used to
store DAG definitions (for pipeline versioning), storing metadata, and passing lightweight data between
tasks via XComs mechanism.

We won't be using example DAGs further, but our own in `dags` directory next to `airflow` directory.
In `airflow.cfg`, set `load_examples = False`, modify `dags_folder` appropriately. Then delete
`airflow.db` (containing saved DAGs), and restart Airflow.

## Intro to building DAGs

DAGs are built using **operators**, which are individual tasks that perform some action. They are
either classes or functions, the latter known
as [TaskFlow API](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/taskflow.html).
The choice is basically personal preference, class-based pipelines are older.

[PythonOperator](https://airflow.apache.org/docs/apache-airflow-providers-standard/stable/operators/python.html)
is used to run arbitrary Python code. In local mode, it shares the environment and libraries with scheduler.

Create `dags` directory next to `airflow` directory (or wherever your `dags_folder` variable in config points to),
and file `01_class_pipeline.py` there. Let's create a DAG and some operators there. Note that it **must** lie in
the global module space to be discovered by Airflow.

```python
import pandas as pd
import requests
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator


def get_data() -> dict:
    print("Fetching data from API")

    # New York temperature in 2025
    url = "https://archive-api.open-meteo.com/v1/archive?latitude=40.7143&longitude=-74.006&start_date=2025-01-01&end_date=2025-12-31&hourly=temperature_2m&timezone=auto"

    resp = requests.get(url)
    resp.raise_for_status()

    data = resp.json()
    data = {
        "time": data["hourly"]["time"],
        "temperature": data["hourly"]["temperature_2m"],
    }
    return data


def transform(data: dict) -> pd.DataFrame:
    df = pd.DataFrame(data)
    df["temperature"] = df["temperature"].clip(lower=-20, upper=50)
    return df


def save_data(df: pd.DataFrame) -> None:
    print("Saving the data")
    df.to_csv("data.csv", index=False)


with DAG(dag_id="weather_data_classes_api"):
    get_data_op = PythonOperator(task_id="get_data", python_callable=get_data)
    transform_op = PythonOperator(
        task_id="transform",
        python_callable=transform,
        op_kwargs={"data": get_data_op.output},
    )
    load_op = PythonOperator(
        task_id="load", python_callable=save_data, op_kwargs={"df": transform_op.output}
    )

    get_data_op >> transform_op >> load_op

```

Important elements:

1. DAG can be
   defined [in a few different ways](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html),
   but a context manager like above is common. All operators defined inside are then
   automatically added to the DAG.
2. Python functions are defined independently of Airflow. They are only passed to
   operator objects as callables, with their arguments passed explicitly via `op_kwargs`.
3. Data is passed between operators via `output` attribute of the previous operator. This is known
   as **XCom (cross-communication)**, serializing and deserializing data in between. By default, it
   uses Airflow database (SQLite here), which is ok for lightweight objects, but not for heavy data.
   We will improve upon this later.
4. Logical flow is defined by chaining operators with `>>` operator. You can also use lists of operators
   as inputs (fan-in, multiple dependencies) and outputs (fan-out, parallel execution).

Run the DAG through UI. Note that the resulting file has been created in the project root - this is the
working directory of Airflow scheduler process. Inspect tasks and their logs, notice how everything
printed in the given task is organized separately per task.

### Exercise 1 (1 point)

Based on the [TaskFlow API tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/taskflow.html#),
rewrite the DAG above to use that instead of class-based operators. Write that in file `02_taskflow_pipeline.py`.

Think which API you prefer. You can use either one for subsequent exercises and homework.

Note that Airflow refreshes DAGs (including getting new ones) by default every 5 minutes. You can control
that with `refresh_interval` variable in configuration, or restart Airflow when you need.

## Scheduling and backfilling

One of the most powerful features of Airflow is **scheduling**, allowing you to run ML pipelines regularly.
Further, you can configure **backfills** to

This is configured primarily via `schedule` DAG parameter, which can generally be used in three modes:

1. **Time-based:** run regularly based on a given timetable, such as
   [cron expression](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/cron.html), e.g.
   retrain ML model once a week.
2. **Assert-aware:** run when the state of an external resource (asset) changes, e.g. new dataset appears in S3 bucket.
3. **Event-based:**
   using [triggers](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/deferring.html),
   they execute code to check condition, e.g. enough new rows appear in a database table.

The time-based option is the most commonly used. It can be easily combined with **backfilling** mechanism to
run the pipeline for historical periods, e.g. to gather historical data from APIs. This mechanism uses
`start_date` parameter to set the first execution date, and optionally `end_date` to set the last one.

Airflow has been designed for ETL processes, which execute **at the end** of the given period. This means that
with daily runs starting from `2025-01-01`, the first run will start on `2025-01-02` midnight. This is known
as **data interval**, and in each run is available as `data_interval_start` and `data_interval_end` variables
(as [Pendulum datetime objects](https://pendulum.eustace.io/docs/#introduction)).
So the run logically scheduled for `data_interval_start` will actually run at `data_interval_end`, covering
that time range.

By "execution date" we mean the **logical date**. Using it as a primary reference point is a good practice
and reduces confusion. Reference it instead of "current time" to avoid further problems. However, in general
this is messy and arguably the most problematic part of Airflow.

See more in
Airflow docs on
[execution date](https://airflow.apache.org/docs/apache-airflow/stable/faq.html#what-does-execution-date-mean),
[data interval](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dag-run.html#data-interval),
or [this StackOverflow answer](https://stackoverflow.com/a/65196624/9472066).

Those values and
other [context variables](https://airflow.apache.org/docs/apache-airflow/stable/templates-ref.html#variables)
are accessed with **kwargs (class-based API) or
through [TaskInstance parameter](https://airflow.apache.org/docs/apache-airflow/stable/templates-ref.html#accessing-airflow-context-variables-from-taskflow-tasks)
(TaskFlow API). You can also pass individual parameters instead of whole `**kwargs` dictionary, which is
often more explicit and a good practice if you use just 1 or 2 arguments.

We will use [TwelveData API](https://twelvedata.com/) now, which has a high 1-minute resolution.
Register there, get an API key, and put it in `.env` file in the project root as `TWELVEDATA_API_KEY`.
Later we will improve on this using Airflow variables.

Copy the code to file `03_scheduling.py`. DAGs start in the "stopped" state by default - turn it on
in the DAGs list panel. Leave it running for a bit, you will see data from subsequent runs in `data.jsonl`.

```python
import datetime
import json
import os

import pendulum

from airflow.providers.standard.operators.python import PythonOperator
from dotenv import load_dotenv
from twelvedata import TDClient

from airflow import DAG

load_dotenv()


def get_data(**kwargs) -> dict:
    logical_date: pendulum.DateTime = kwargs["logical_date"]

    td = TDClient(apikey=os.environ["TWELVEDATA_API_KEY"])

    ts = td.exchange_rate(symbol="USD/EUR", date=logical_date.isoformat())
    data = ts.as_json()
    return data


def save_data(data: dict) -> None:
    print("Saving the data")

    if not data:
        raise ValueError("No data received")

    with open("data.jsonl", "a+") as file:
        file.write(json.dumps(data))
        file.write("\n")


with DAG(
        dag_id="scheduling_dataset_gathering",
        schedule=datetime.timedelta(minutes=1),
) as dag:
    get_data_op = PythonOperator(task_id="get_data", python_callable=get_data)
    save_data_op = PythonOperator(
        task_id="save_data",
        python_callable=save_data,
        op_kwargs={"data": get_data_op.output},
    )

    get_data_op >> save_data_op

```

[Catchup mechanism](https://airflow.apache.org/docs/apache-airflow/3.1.5/core-concepts/dag-run.html#catchup)
in Airflow automatically runs the DAG for previous periods,
while [backfill mechanism](https://airflow.apache.org/docs/apache-airflow/3.1.5/core-concepts/dag-run.html#backfill)
runs it for given date range. This allows easy gathering of historical data for ML datasets. See
[this blog post](https://medium.com/@seilylook95/understanding-the-difference-between-airflows-backfill-and-catchup-cf6e830588b8)
for a detailed explanation of differences between backfilling and catchup.

To use catchup, set `catchup=True` and `start_date` parameters in DAG definition. When you run the DAG,
it will run with the given schedule from that date until current time, or `end_date` if you also set it.

You can run backfills manually from the UI, or by using Airflow CLI, like:

```bash
airflow backfill create \
  --dag-id backfilling_dataset_gathering \
  --from-date 2025-01-01 \
  --to-date 2025-01-31
```

### Exercise 2 (1 point)

Using [Open Meteo historical forecast API](https://open-meteo.com/en/docs/historical-forecast-api), gather
daily New York weather forecast data for January 2025: min and max temperature
(you can go over the last date).
Remember to set a time zone in the API request. Save results as a CSV file. Use catchup in Airflow,
which will automatically trigger the missing runs,
and [delta time interval](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/timetable.html#crondataintervaltimetable)
to run it every 7 days. Save the script as `04_backfilling.py`.

Take care to avoid overlapping dates, as that would duplicate rows in the resulting dataset. Remember
that you have `logical_date` variable available in `**kwargs`.

Such data is used, for example, to augment demand forecasting with weather information. Since for production
deployment we would use forecasts, here we gather historical forecasts. We simulate the situation where
we want to forecast every weekend to prepare for the next week. Thus, every week, we gather daily
forecasts for the next week.

## Practical Airflow deployment

So far, we have been working with local, non-isolated Airflow installation. Now let's properly deploy
it as a set of services, using Docker
Compose. [Documentation recommends using Kubernetes](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)
for full-scale production deployments, but Docker Compose works well locally.

We will also switch the executor from local, running tasks in scheduler subprocesses,
to [Celery executor](https://airflow.apache.org/docs/apache-airflow-providers-celery/stable/celery_executor.html).
It uses [Celery task queue](https://docs.celeryq.dev/en/stable/) and separate workers (Python processes)
to run tasks, typically backed by Redis or RabbitMQ for message queueing.

A typical Airflow deployment stack includes:

1. Airflow API web server, scheduler, DAG processor, and triggerer
2. Postgres database for metadata
3. Celery + Redis or RabbitMQ
4. Airflow worker, based on Celery executor

Create file `compose.yaml` in project root directory and copy the contents below there. This sets up
the configuration described above. [Valkey](https://valkey.io/) is used as an open source Redis alternative,
and it's also [much more scalable](https://valkey.io/blog/unlock-one-million-rps/) thanks to multithreading.
[YAML anchors](https://support.atlassian.com/bitbucket-cloud/docs/yaml-anchors/) are used to share
common configuration between services.

```yaml
# common configuration with anchor
x-airflow-common: &airflow-common
  build:
    context: .
    args:
      AIRFLOW_IMAGE_NAME: apache/airflow:3.1.5-python3.11
  env_file:
    - .env
  environment: &airflow-common-env
    # DB and Celery executor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    AIRFLOW__CORE__EXECUTOR: CeleryExecutor
    AIRFLOW__CELERY__BROKER_URL: redis://:@valkey:6379/0
    AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@postgres/airflow

    # a few general settings
    AIRFLOW_CONFIG: '/opt/airflow/config/airflow.cfg'
    AIRFLOW__CORE__AUTH_MANAGER: airflow.providers.fab.auth_manager.fab_auth_manager.FabAuthManager
    AIRFLOW__CORE__EXECUTION_API_SERVER_URL: "http://airflow-apiserver:8080/execution/"
    AIRFLOW__CORE__LOAD_EXAMPLES: "false"
    AIRFLOW__SCHEDULER__ENABLE_HEALTH_CHECK: "true"
    AIRFLOW__DAG_PROCESSOR__REFRESH_INTERVAL: "10"

    # fake security for local setup, load this from .env or remote secret backend in production!
    AIRFLOW__CORE__FERNET_KEY: "Jj8Zk4n4Y3aP0uR2Zc1JvZ8aZxqf4o8WQ1l8J7nM2Kk"
    AIRFLOW__API__SECRET_KEY: "sv3N1i8Wwu9W6IsUvcDjCg=="
    AIRFLOW__API_AUTH__JWT_SECRET: "sv3N1i8Wwu9W6IsUvcDjCg=="

    # use uv + cache
    AIRFLOW__STANDARD__VENV_INSTALL_METHOD: "uv"
    UV_CACHE_DIR: /opt/airflow/.uv-cache

  volumes:
    # volumes to sync local environment and containers
    - ./dags:/opt/airflow/dags
    - ./logs:/opt/airflow/logs
    - ./plugins:/opt/airflow/plugins
    - ./config:/opt/airflow/config
    - ./.uv_cache:/opt/airflow/.uv-cache

  user: "${AIRFLOW_UID:-50000}:0"
  depends_on:
    postgres:
      condition: service_healthy
    valkey:
      condition: service_healthy

services:
  postgres:
    image: postgres:17
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-db-volume:/var/lib/postgresql/data
    healthcheck:
      test: [ "CMD", "pg_isready", "-U", "airflow" ]
      interval: 10s
      retries: 5
    restart: always

  valkey:
    image: valkey/valkey:9
    healthcheck:
      test: [ "CMD", "valkey-cli", "ping" ]
    restart: always

  airflow-init:
    <<: *airflow-common
    environment:
      <<: *airflow-common-env
      _AIRFLOW_DB_MIGRATE: 'true'
      _AIRFLOW_WWW_USER_CREATE: 'true'
      _AIRFLOW_WWW_USER_USERNAME: airflow
      _AIRFLOW_WWW_USER_PASSWORD: airflow
      _PIP_ADDITIONAL_REQUIREMENTS: ''
    user: "0:0"

  airflow-apiserver:
    <<: *airflow-common
    command: api-server
    ports:
      - "8080:8080"
    depends_on:
      airflow-init:
        condition: service_completed_successfully
    restart: always

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
    depends_on:
      airflow-init:
        condition: service_completed_successfully
    restart: always

  airflow-worker:
    <<: *airflow-common
    command: celery worker
    environment:
      <<: *airflow-common-env
      DUMB_INIT_SETSID: "0"
      AIRFLOW__CELERY__WORKER_CONCURRENCY: "4"
    depends_on:
      airflow-apiserver:
        condition: service_started
      airflow-init:
        condition: service_completed_successfully
    restart: always

  airflow-triggerer:
    <<: *airflow-common
    command: triggerer
    depends_on:
      airflow-init:
        condition: service_completed_successfully
    restart: always

  airflow-dag-processor:
    <<: *airflow-common
    command: dag-processor
    depends_on:
      airflow-init:
        condition: service_completed_successfully
    restart: always

volumes:
  postgres-db-volume:
  uv-cache:

```

The code above uses Dockerfile to customize Airflow image with additional libraries common
for all workers. Create `Dockerfile` file and copy the following context.

```dockerfile
ARG AIRFLOW_IMAGE_NAME=apache/airflow:3.1.5-python3.11
FROM ${AIRFLOW_IMAGE_NAME}

USER root
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /opt/airflow
COPY pyproject.toml uv.lock ./
RUN uv pip install --system pyproject.toml --group airflow_common

USER airflow

```

With this new setup, we can simplify `pyproject.toml` and dynamically install dependencies in
DAG. Now the only things the workers share by default are libraries from `airflow_common` group,
`requests` in the example below. It's a good practice to minimize the number of common dependencies,
as this reduces conflicts and memory usage. We also
need [Airflow extras](https://airflow.apache.org/docs/apache-airflow/stable/extra-packages-ref.html)
to make it work with Postgres and Celery, as well as to use uv to install dependencies in Python
venv operators.

```toml
[project]
name = "MLOps_course_AGH_lab_02"
version = "1.0.0"
description = ""
readme = "README.md"

requires-python = "==3.11.*"

dependencies = [
    "apache-airflow[celery,postgres,standard,uv]==3.1.5",
    "twelvedata",
]

[dependency-groups]
dev = [
    "ruff"
]

airflow_common = [
    "apache-airflow[celery,postgres,standard,uv]==3.1.5",
    "requests",
]

```

Before we run Airflow, we also need to make sure that Docker can both read and write files
in its volumes. Run the commands below to do so.

```bash
mkdir -p dags logs plugins config .uv_cache
sudo chmod -R 777 logs dags plugins config .uv_cache
```

Now when you run `docker compose up`, a few things happen:

- all services go up
- their healthchecks are used to verify status
- Airflow migrates the DB if necessary and start the webserver at `http://localhost:8080`
- you log in with `airflow` user and password
- all workers run as Python processes in Celery, sharing the same environment with `requests` installed

### Exercise 3 (1 point)

Create DAG `05_scheduling_with_venvs.py`, installing the `twelvedata` library dynamically in the
appropriate task.
Use [PythonVirtualenvOperator](https://airflow.apache.org/docs/apache-airflow-providers-standard/stable/operators/python.html#pythonvirtualenvoperator)
to do so. Note that imports using dynamically installed libraries need to be inside the function,
not at the top level.

**Warning:** note that combining PythonVirtualenvOperator with Airflow context
variables [is highly nontrivial](https://github.com/apache/airflow/issues/20974),
since Airflow needs to pass data between two different Python interpreters.

Recommended setup is:

- always explicitly use `cloudpickle` serializer
- always add `pendulum` and `lazy_object_proxy` to requirements whenever you need access to Airflow context
- pass `data_interval_start` parameter, not whole `**kwargs`

## Object storage XCom backend

When you return something from an Airflow function, it is stored
using [XComs (cross-communications) mechanism](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/xcoms.html).
It serializes and deserializes the data automatically, taking care of its location for you. By default,
it stores it in the Airflow metadata database, e.g.
as [Postgres binary blobs](https://www.postgresql.org/docs/7.4/jdbc-binary-data.html).
However, this is inefficient for large objects and may even crash the database.

You can switch
to [object storage XComs backend](https://airflow.apache.org/docs/apache-airflow-providers-common-io/stable/xcom_backend.html)
to solve this problem. It uses object storage, such as AWS S3, for storing larger objects,
and DB for smaller ones, like single Python variables.

For development purposes, [LocalStack](https://github.com/localstack/localstack) can be used to
emulate AWS services. Install [LocalStack CLI](https://docs.localstack.cloud/aws/getting-started/installation/)
and add its service to the Docker Compose setup:

```yaml
  localstack:
    image: localstack/localstack
    ports:
      - "127.0.0.1:4566:4566"            # LocalStack Gateway
      - "127.0.0.1:4510-4559:4510-4559"  # external services port range
    environment:
      SERVICES: "s3"  # just S3 service
    volumes:
      - localstack_data:/var/lib/localstack
      - "/var/run/docker.sock:/var/run/docker.sock"
      - ./localstack-init:/etc/localstack/init
    healthcheck:
      test: [ "CMD", "curl", "-fsS", "http://localhost:4566/_localstack/health" ]
      interval: 10s
      retries: 10
    restart: always
```

Further:

1. Create `localstack-init/ready.d/01-create-buckets.sh` file
   with [initialization hook](https://docs.localstack.cloud/aws/capabilities/config/initialization-hooks/),
   content:

```bash
#!/bin/sh
set -e

awslocal s3 mb s3://airflow-xcom || true

```

2. Make sure that it's executable:

```bash
chmod +x localstack-init/ready.d/01-create-buckets.sh
```

3. Add `localstack_data` to the global list of volumes in `compose.yaml`.
4. Append `localstack` to `depends_on` section of common Airflow services.

Now, when you rerun Docker Compose, LocalStack will have `airflow-xcom` S3 bucket ready.

To interact with LocalStack, you use regular AWS CLI commands. However, you must point to the
appropriate endpoint, `http://localhost:4566` in this case (since we expose port 4566).

For example, listing buckets:

```bash
aws --endpoint-url=http://localhost:4566 s3 ls
```

Listing files in `airflow-xcom` bucket:

```bash
aws --endpoint-url=http://localhost:4566 s3 ls s3://airflow-xcom
```

At this point, we have a fully functioning local S3. We could
use [Airflow S3 plugin](https://airflow.apache.org/docs/apache-airflow-providers-amazon/stable/operators/s3/s3.html)
with it, or combine
with [Airflow s3fs plugin](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/objectstorage.html) to access
S3 identically to local storage. Those options are useful in themselves, but now we will plug in this S3 bucket as
XCom backend.

This is controlled by environment variables. Add them to the Airflow common env in `compose.yaml`:

```yaml
    # set up LocalStack S3 as XCom backend
    AIRFLOW__CORE__XCOM_BACKEND: "airflow.providers.common.io.xcom.backend.XComObjectStorageBackend"
    AIRFLOW__COMMON_IO__XCOM_OBJECTSTORAGE_PATH: "s3://aws_default@airflow-xcom/xcom"
    AIRFLOW__COMMON_IO__XCOM_OBJECTSTORAGE_THRESHOLD: "0"  # threshold 0 = all files are stored in S3

    # URL-encoded fake AWS connection, we must set endpoint_url for LocalStack
    AIRFLOW_CONN_AWS_DEFAULT: 'aws://test:test@?endpoint_url=http%3A%2F%2Flocalstack%3A4566&region_name=us-east-1'

```

Further, we need additional dependencies in `pyproject.toml`:

- `s3fs` as an Airflow extra (written in `[]` after package name)
- `apache-airflow-providers-amazon`

[Providers in Airflow](https://airflow.apache.org/docs/apache-airflow-providers/) are plugins
for external services, providing additional operators and capabilities. They are configured with
[Connections](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/connections.html#), which
are the focus of the next section.

Run any of the previous DAGs again and list files in `airflow-xcom` bucket. You should

#### Exercise 4 (1 point)

1. Modify LocalStack to create another bucket `weather-data`.
2. Create DAG `06_s3_integration.py` based on `01_class_pipeline.py`.
   Use [s3fs Airflow integration](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/objectstorage.html)
   to store results in S3.
3. Save results in S3 bucket as CSV. Note that the logical date should be part of the file name.

### Connections, hooks, variables

[Connections](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/connections.html#)
are used by Airflow to standardize connecting to external services. Those can be, e.g., API keys,
AWS / GCP / Azure credentials, or database connection strings.

Above, we defined
the [AWS connection](https://airflow.apache.org/docs/apache-airflow-providers-amazon/stable/connections/aws.html),
with value starting with `aws://`:

```yaml
AIRFLOW_CONN_AWS_DEFAULT: 'aws://test:test@?endpoint_url=http%3A%2F%2Flocalstack%3A4566&region_name=us-east-1'
```

Such connections are known
as [environment variable connections](https://airflow.apache.org/docs/apache-airflow/stable/howto/connection.html#storing-connections-in-environment-variables)
and are defined as `AIRFLOW_CONN_{connection_name}`.
Unfortunately, they do not show up on Airflow UI, since they are dynamically computed. You can load their
values from `.env` files
or [secret backends](https://airflow.apache.org/docs/apache-airflow/stable/security/secrets/secrets-backend/index.html),
such as AWS SSM Parameter Store.

You can also set up
connections [manually in the UI](https://airflow.apache.org/docs/apache-airflow/stable/howto/connection.html#storing-connections-in-the-database).
They are also persistent and securely stored in the
Airflow database, so for long-running Airflow installations they are a viable option.

A similar and related concept
are [Variables](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/variables.html).
They are key-value pairs that can be used to store arbitrary data in Airflow metadata DB, and also
can be encrypted using the Fernet key. You
can [set them via UI](https://airflow.apache.org/docs/apache-airflow/stable/howto/variable.html),
or using environment variables like `AIRFLOW_VAR_{VARIABLE_NAME}`. They are handy for securely
storing API keys, for example. This can also be directly connected to the secrets backend, same
as connections. This way, Airflow offers an easy-to-use abstraction layer.

Example variable usage:

```python
from airflow.sdk import Variable


def make_llm_embeddings(texts):
    openai_api_key = Variable.get("OPENAI_API_KEY")
    ...

```

[Hooks](https://www.astronomer.io/docs/learn/what-is-a-hook) are an abstraction over services
implemented in Airflow, directly using connections, implementing common operations. Using them
is entirely optional, but they can simplify the code and make it more readable. For
example, [S3Hook](https://airflow.apache.org/docs/apache-airflow/1.10.6/_api/airflow/hooks/S3_hook/index.html)
simplifies S3 usage,
while [PostgresHook](https://airflow.apache.org/docs/apache-airflow-providers-postgres/stable/_api/airflow/providers/postgres/hooks/postgres/index.html#airflow.providers.postgres.hooks.postgres.PostgresHook)
wraps interactions with Postgres.

```python
from airflow.providers.postgres.hooks.postgres import PostgresHook

POSTGRES_CONN_ID = "postgres_default"


def load_dataset_from_table():
    pg_hook = PostgresHook.get_hook(POSTGRES_CONN_ID)
    df = pg_hook.get_df("SELECT * from historical_ratings")
    ...

```

#### Exercise 5 (1 point)

1. Add variable `TWELVEDATA_API_KEY` to Airflow, using `.env` file.
2. Add a second Postgres database to Docker Compose, emulating standalone storage DB.
   Create table `exchange_rates` there, with columns `symbol` (string) and `rate` (float).
   [Init hook](https://gist.github.com/onjin/2dd3cc52ef79069de1faa2dfd456c945) may be useful.
3. Create a connection to this database in Airflow UI.

Based on this configuration, create a DAG `07_connections_and_variables.py` based on
`05_scheduling_with_venvs.py` that:

1. Uses Airflow variable instead of a raw environment variable. Pass it as parameter
   to the operator - `PythonVirtualenvOperator` is isolated, we cannot read global Airflow
   config there.
2. Uses `PostgresHook` to put the data into the database table, instead of writing to file.
