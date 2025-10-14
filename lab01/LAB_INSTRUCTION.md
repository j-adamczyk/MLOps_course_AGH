# Laboratory 1 instruction

## Introduction

In this lab, we will build a production-ready machine learning application. Step by step, we will
introduce best practices and basic tools that are used throughout any MLOps project. At the end,
you will have local server providing ML model predictions.

We will cover:
1. Code versioning and Git
2. Dependency management
3. Pre-commit hooks
4. Configuration, environment variables management
5. Secrets management
6. Testing
7. FastAPI webserver
8. Serving ML model
9. Containerization, Docker, Docker Compose

Instructions:
- do tasks in order, they have been designed to build upon prior steps
- write clean code, descriptive variable and function names, use comments where necessary etc.
- send your resulting application code either as .zip file or GitHub public repository link

---

## 1. Code versioning and Git  (0.5 points)

Code versioning is a crucial part of software development. It allows you to keep track of changes,
navigate through the source code history, and collaborate with other developers. We will use Git
for version control and GitHub as Git repository hosting.

If you don't have a GitHub account, create one.

1. Create a new repository on GitHub, with any reasonable name. Initialize it with README,
   `.gitignore` (Python template) and MIT license.

2. Clone the repository to your local machine.
```bash
git clone <repository-url>
```

3. Modify README.md, e.g. change title. Then, add it to the local **staging area** with `git add`.
   Then, commit the changes with `git commit`. Note that this commit is still on your machine at
   this point.
```bash
git add README.md
git commit -m "Edit README.md file"
```

4. Push the changes to the remote repository with `git push`, sending the commit and code changes.
   Note that this command assumes the `master` branch to be the default one. Change this to `main`
   if needed, you can configure this default in your GitHub account settings.
```bash
git push -u origin master
```

For open source projects, permissive license like MIT, BSD or Apache are preferred. For
private, proprietary code, you generally don't need one, but you have to avoid using any
GPL-licensed libraries.

---

## 2. Dependency management (0.5 points)

Dependency management is an important aspect of software development. 
It helps you manage the project dependencies, set constraints on allowed library versions,
update them in a controlled way, and ensures that the application runs smoothly across
different environments.

`uv` is an extremely fast Python dependency and project manager, written in Rust.
It allows you to switch between Python versions, resolve and pin dependencies, and install them
blazingly fast. If you don't have it installed yet, follow [the official documentation](https://docs.astral.sh/uv/)
for your operating systems.

Verify the installation by running:
```bash
uv --version
```

Let's check the available python versions and implementations that can be installed using `uv`:
```bash
uv python list --all-platforms
```

As you can see, there is pretty long list of available python versions. `uv` is mostly concerned
with CPython, the official implementation written in C. Alternatives include e.g.:
- Cython, with custom compiled extensions, often used for ML algorithms implementation
- PyPy, based on JIT compiler
- Anaconda, proprietary, focused on data science

In this lab, we will use Python 3.12, the standard CPython implementation.

To initialize the Python project in a current directory, run:
```bash
uv venv --python 3.12
uv init
```
Those commands will create the **virtual environment**, separating Python instance and its
dependencies in the current projects from the main system one. It will also create files:
- `pyproject.toml` - used for specifying dependencies
- `.python-version` - stating Python version used

To activate it, on Linux and macOS run `source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows.

Verify the Python version by running the following command:
```bash
python --version
```
The output should point to the virtual environment directory, inside the current project directory.

Verify the virtual environment by running the following command:
- Linux / macOS: `which python`
- Windows: `where python`

`pyproject.toml` contains the project metadata and dependencies. See [uv documentation](https://docs.astral.sh/uv/guides/projects/#pyprojecttoml)
and [official pyproject.toml guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) for details.
You can add the dependencies using the following command:
```bash
uv add <package-name>
```
The dependencies will be added to the `pyproject.toml` file and installed in the virtual environment.
You can notice that a new file `uv.lock` has been created in the project directory. 
It contains the exact versions of the packages you stated, as well as their own dependencies resolved
by `uv`. It automatically installs them when added.

Alternatively, you can also write dependencies manually. To resolve dependencies, call `uv lock`. To
install them, run `uv sync`.

Commit and push `pyproject.toml` and `uv.lock` to the GitHub repository. This will ensure full
reproducibility for other developers.

---

## 3. Pre-commit hooks (0.5 points)

Pre-commit hooks are scripts that run automatically when you make a new commit. They actually run
just before it, and prevent the commit if any error is raised. It is a good practice to run a few
lightweight hooks, e.g. formatting, linting, or code analysis.

We will use following pre-commit hooks:
* [Ruff](https://docs.astral.sh/ruff/) - code linter and formatter
* [Xenon](https://github.com/yunojuno/pre-commit-xenon) - code complexity checker, based on [Radon](https://github.com/rubik/radon/)

1. Install the pre-commit package using uv:
```bash
uv add pre-commit
```

2. Create `.pre-commit-config.yaml` file in the project directory. It contains configuration
   of pre-commit hooks used. Add Ruff now:
```yaml
repos:
   - repo: https://github.com/astral-sh/ruff-pre-commit
     rev: v0.9.5
     hooks:
        - id: ruff  # Linter
          args: [--fix]  # Ensure fixes are applied
        - id: ruff-format  # Formatter
```
Those are actually two separate hooks - linter and formatter.

3. Pre-commit hooks are a separate mechanism from the `uv` and your exact project. They are
   also installed separately, with command:
```bash
pre-commit install
```

4. Add the `.pre-commit-config.yaml` to staging area and commit changes. Now the hook will
   run before every commit. If you run Git in the terminal, you will see the outputs there.
   In PyCharm, they show up in `Git` tab, and in similar places in other IDEs. To run hooks
   manually, run:
```bash
pre-commit run
```
Many hooks track what files changed since they were run, and only run on those for speed.
You can use `pre-commit run --all-files` to force checking all files.

5. Add the Xenon pre-commit hook, [based on its documentation](https://github.com/rubik/xenon).
   Write its configuration below Ruff in `.pre-commit-config.yaml`. To install it and verify
   if it's working, run:
```bash
pre-commit clean
pre-commit install
pre-commit run --all-files
```

6. Add the `.pre-commit-config.yaml` with Xenon to the staging area and commit the changes.

---

## 4. Environment variables management (0.5 points)

**Environment variable** is a named variable, living inside your process. They can be used
to pass configuration to processes and programs, which read them using system utilities. Think
of them like system variables, living not in your program, but in the process itself. They
have string names and values, with names traditionally written in UPPER_CASE.

For environment variables management, we will use:
* [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
* [.env files](https://hexdocs.pm/dotenvy/0.2.0/dotenv-file-format.html)

1. Install the `pydantic-settings` package using `uv`. It is used for a Pythonic, class-based
   handling of configuration. In particular, it can automatically read environment variables
   and presents them as attributes.

2. Create `settings.py` file in the project directory. Define the settings class:
```python
# settings.py
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    ENVIRONMENT: str
    APP_NAME: str

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, value):
       ... # implement me!
       # prepare validator that will check whether the value of ENVIRONMENT is in (dev, test, prod) 
       return value
```
Fill the `validate_environment` function, checking the parsed environment value. It should be one
of: "dev", "test", "prod". If it's alright, return it, otherwise raise `ValueError`. Pydantic will
detect this and raise `ValidationError` with readable message. Use the
[documentation examples](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) as necessary.

3. `.env` (read: "dotenv") is a very simple format for configuration files, popular for creating
   environment variables. They are loaded into the process, and later read as configuration. Typically,
   we want to have separate environments for different stages of development. They have different
   configuration, e.g. testing database has different address and password from the production one.
   Create new folder `config` in main project directory, and store there 3 new files, each with `ENVIRONMENT` and `APP_NAME` variables:
   - `.env.dev`
   - `.env.test`
   - `.env.prod`

4. We will now prepare the `main.py` file. `.env` file needs to be loaded in the environment, which
   can be done in a few ways, e.g. with `bash`, frameworks, or simply with Python. We will use the
   latter here.
   - add `python-dotenv` package with `uv`, it can load `.env` files into your process environment
   - copy the code below into `main.py`
   - fill `export_envs` function to select `.env` file based on provided environment argument
     and load it into environment
   - check with `uv run python main.py --environment dev` (`uv run` ensures that the command runs
     inside the environment)
   - change environment value in `.env.dev` to "xyz" and run the script again, it should raise
     an error

```python
import os
import argparse
from dotenv import load_dotenv
from settings import Settings


def export_envs(environment: str = "dev") -> None:
    ... # implement me!


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load environment variables from specified.env file.")
    parser.add_argument("--environment", type=str, default="dev", help="The environment to load (dev, test, prod)")
    args = parser.parse_args()

    export_envs(args.environment) 
    
    settings = Settings()
    
    print("APP_NAME: ", settings.APP_NAME)
    print("ENVIRONMENT: ", settings.ENVIRONMENT)
```

---

## 5. Secrets management (0.5 points)

**Secrets** are sensitive variables like passwords, API keys and other information that should
be secured. Regular configuration elements (e.g. environment, app name) can be kept in plaintext files.
Secrets require **encryption** and secure storage as files. For usage in the actual process, we need
to decrypt them and export as environment variables. We will use [sops package](https://github.com/getsops/sops)
for this.

1. Install the sops, following [the documentation](https://github.com/getsops/sops).

2. To encrypt and decrypt files, you'll need a [GPG key](https://confluence.atlassian.com/bitbucketserver/using-gpg-keys-913477014.html).
   Generate one using the following command:
```bash
gpg --gen-key
```

3. Once you have the GPG key, you can list the keys using the following command:
```bash
gpg --list-secret-keys --keyid-format LONG
```

It should look like: `sec rsa4096/0F861E6EC136294F` and YOUR_KEY_ID is:  `0F861E6EC136294F `

4. GPG uses private and public key pair. Public key is chared with others and used for encryption. Private
   key should **never** be shared, however! For now, we will export the public key to a file. In the
   command below, replace `YOUR_KEY_ID` with your actual key ID.
```bash
gpg --armor --export YOUR_KEY_ID > public_key.asc
```

5. We will use [sops](https://github.com/getsops/sops) for encrypting secrets, using the GPG key
   for actual encryption. Create `.sops.yaml` configuration file in your project root, and specify
   the GPG key ID inside it:
```yaml
creation_rules:
  - pgp: '<YOUR_KEY_ID>'
```

6. SOPS can encrypt any YAML file, which is very useful for keeping secrets. Such encrypted
   file can be put in the version control system and shared. Create `secrets.yaml` file and put
   any key and value there, e.g. a fake API key.
7. Encrypt `secrets.yaml` with SOPS:
```bash
sops --encrypt --in-place secrets.yaml 
```
Check the file contents. It is a text format, but encrypted. Thus, you can safely commit it in Git.

8. Now decrypt the file and export variables
```bash
sops --decrypt --in-place secrets.yaml
```

If the key doesn't work (but it's ID is listed), maybe gpg can't find your private key in it's directory. Try:
```
echo 'export GPG_TTY=$(tty)' >> ~/.bashrc
```
And restart terminal. If this still doesn't work, try to reset everything:
```
rm -rf ~/.gnupg
mkdir -p ~/.gnupg
echo "pinentry-program /usr/bin/pinentry-curses" > ~/.gnupg/gpg-agent.conf
gpgconf --kill gpg-agent
chmod 700 ~/.gnupg
```
And then generate key from the start (point 2.)

9. Update the application:
   - add `pyyaml` with `uv`, it can read YAML files ([tutorial](https://python.land/data-processing/python-yaml))
   - load decrypted file as environment variables, using `pyyaml` and `os` modules
   - modify `Settings` class to also load the environment variable with secret

10. Run `main.py` and check the value of the secret.
11. Commit the changes.

---

## 6. Testing (0.5 points)

Testing is an important aspect of software development. It helps to ensure that the application
works as expected and there are no obvious bugs. We will use `pytest` as our testing framework.

1. Install the `pytest` and `pytest-dotenv` packages using `uv`.
2. Prepare `.env.test` file containing the variables that should be loaded to `Settings` object while tests run.
   It should have the same keys as expected production configuration, but fake values.
3. Prepare `pytest.ini` file with the configuration for `pytest`. We could also put this in `pyproject.toml`,
   this depends on personal preference.
```toml
[pytest]
env_files =
    ./config/.env.test
```
4. Create `tests` directory, where we will implement tests.
5. Implement the basic test for settings. It should check if settings are loaded correctly
   and contain all the expected values.
6. Write test cases for the application that cover the functionality:
7. Run the tests using the following command:
```bash
uv run pytest tests -rP
```
If you encountered a problem where pytest can't load `src` module, try:
```
pythonpath = .
```
8. Commit the changes.

---

## 7. Webserver and application setup (0.5 points)

In machine learning, [FastAPI](https://fastapi.tiangolo.com/) is the most popular choice
for implementing the webserver. It is a modern, high-performance web framework for building
Python APIs, based on [Starlette](https://www.starlette.io/) as ASGI framework (handling web-related
stuff, in short), and [Pydantic](https://docs.pydantic.dev/latest/) for data validation. Its
popularity stems from ease of usage and very Pythonic code style.

Some key features include:
- automatic interactive API documentation (Swagger & ReDoc)
- fast execution (thanks to Starlette & asynchronous capabilities)
- type hints support for better editor autocompletion and error detection
- data validation built in (Pydantic)

If you want to learn more about those tools and their purpose inside FastAPI, see e.g.
[this great explanation](https://tonybaloney.github.io/posts/fine-tuning-wsgi-and-asgi-applications.html),
[this tutorial](https://dev.to/kfir-g/understanding-fastapi-fundamentals-a-guide-to-fastapi-uvicorn-starlette-swagger-ui-and-pydantic-2fp7)
or [this blog post](https://jieunjeon.com/fastapi-uvicorn-and-asgi-the-trio-powering-modern-web-apps/).

1. Install the `fastapi` and `uvicorn` packages using `uv`.
2. Create `app.py` file in the project directory. We will create a simple the FastAPI application
   with the code below. For now, it will contain two simple endpoints (routes):
   * `/` (root) - returns a welcome message
   * `/health` - returns the health status of the application, used by many orchestrators and
     management systems to verify if server is working
```python
from fastapi import FastAPI


app = FastAPI()


@app.get("/")
def welcome_root():
    return {"message": "Welcome to the ML API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
```
3. Run the actual application server, exposed on port 8000:
```bash
uv run uvicorn app:app --reload --port 8000
```
4. Open the browser and go to http://localhost:8000 to see the welcome message.
5. Navigate to http://localhost:8000/health to invoke `/health` endpoint and see
   the health status of the application.
6. Write tests for the application routes. Verify if selected routes return the expected responses.
7. Commit the changes.

---

## 8. Serving machine learning model (0.5 points)

We have a working server, but no machine learning so far, so let's add a model,
implemented in [scikit-learn](https://scikit-learn.org/stable/).

1. Install the `scikit-learn` package using `uv`.
2. Before we perform inference (serve a model), we need the actual model:
   - create `training.py` file
   - `load_data()` function - loads and returns the
     [iris dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
   - `train_model()` - trains any simple classification model and returns it
   - `save_model()` - saves the trained model to file using `joblib`
3. To serve the model, we will need to load the model and make predictions:
   - create `inference.py` file
   - `load_model()` - loads the trained model from file
   - `predict()` - uses the model to make predictions, returning predicted class as a string
4. In FastAPI, Pydantic is used for validating requests and responses. It automatically parses
   JSON data as Python objects and verifies keys, values, and their types. Pydantic uses classes
   and types to define the format of request and response. Create new directory and file `api/models/iris.py`,
   and define API request and response models there:
```python
from pydantic import BaseModel


class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictResponse(BaseModel):
    prediction: str
```
5. Load the model as a global variable in `app.py`. This global code runs once, on the application initialization,
   so the model will be loaded and kept in memory.
6. Create a new endpoint (route) in `app.py` file that accepts the input data for our model and returns the prediction:
```python
@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    prediction = model.predict(request.model_dump())
    return PredictResponse(prediction=prediction)
```
7. Run the app and verify that the new route works as expected.
8. Commit the changes.

---

## 9. Containerization (1 point)

For containerization, we will use [Docker](https://www.docker.com/). It is a platform that allows you to
build, ship, and run applications in isolated containers. They are lightweight, portable, and self-sufficient
environments that contain everything needed to run the application. The idea is basically:
- "It works on my machine"
- "So we will ship your machine"

Docker **images** are used to create running **containers**. To define the image, we use **Dockerfile**,
with its own specific format. It starts with a **base image**, which is the operating system and its basic
dependencies. Then, you add **layers**, i.e. any commands and changes to that image. You can install
additional packages, copy project files like code or models inside the container, configure environment,
and finally run the application.

We will also use [Docker Compose](https://docs.docker.com/compose/). It is a container orchestrator, i.e.
it manages multi-container environments. However, it's also very useful even for a single container applications,
as it uses a very readable YAML configuration file.

1. Make sure you have Docker installed on your local machine, e.g. run `docker ps`.
2. Let's containerize our application, i.e. put code inside an isolated container.
   Create a Dockerfile (literally `Dockerfile` text file) in the project directory 
   and define the Docker image for the application. Change the directory names if necessary.
```dockerfile
# Dockerfile

# Use a minimal Python image as the base
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    curl libsnappy-dev make gcc g++ libc6-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Copy only dependency files first (to leverage caching)
COPY lab/pyproject.toml lab/uv.lock ./

# make uv work in system's python
ENV UV_PROJECT_ENVIRONMENT=/usr/local
# Install project dependencies using uv
RUN uv sync

# Copy the rest of the application code
COPY lab . 

# Expose the application port
EXPOSE 8000

# Run the application with uv
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

3. Build the Docker image with command below. This will download base image, apply all layers, and save the
   resulting image in your local Docker registry. `-t` or `--tag` adds a named tag to the image, allowing
   you to use that tag instead of randomly generated ID.
```bash
docker build -t ml-app .
```
To check the images available, run `docker images`.

4. Run the Docker container. We select the image by providing a tag. We also **expose** a port - by default
   Docker containers run in a complete network isolation, but here we want to access the server running in
   the container from our host machine.
```bash
docker run --rm -p 8000:8000 ml-app
```
To see the running containers, run `docker ps`.

5. Open the browser and navigate to http://localhost:8000. Verify that the application is running.
6. Commit the changes.
7. To stop and turn off the container, you can use either container ID or its name. You can check them
   with `docker ps`.
```bash
docker stop <container-id>
```

Last thing will be setting up Docker Compose. Here, we have only a single container with our server,
but we could also add databases (e.g. Postgres, Redis) or frontend (e.g. React). We can also define
inter-container dependencies, such as "application is ready when all containers are up" or "run frontend
after backend containers are up". Either publicly available images can be used, or our custom ones,
possibly with multiple Dockerfiles defining distinct services.

Docker Compose uses YAML format for configuration. It is very readable, and we can put options there
that we previously passed manually in the commandline to Docker. This makes it very useful even
for single container applications, as we don't have to type everything each time.

1. Make sure you have Docker Compose installed, e.g. run `docker compose ps`
2. Create `docker-compose.yaml` file:
```yaml
version: '3.8'

services:
  ml-app:
    build: .
    ports:
      - "8000:8000"
```
3. Run the application:
```bash
docker compose up
```
4. Open the browser and navigate to http://localhost:8000. Verify that the application is running.
5. Turn off with:
```bash
docker compose down
```
