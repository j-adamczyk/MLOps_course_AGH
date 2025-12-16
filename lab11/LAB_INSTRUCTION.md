# Lab 11 - Deployment & CI/CD

## 1. Introduction

This lab will cover deployment of ML models for inference. Models are optimized
for inference with tools like ONNX, and deployment processes are automated with
CI/CD pipelines. This results in efficient and reliable model updates.

We recommend using own local environment. However, if you have problems or would
like to try something new, GitHub Codespaces can also be used.

---

### Why do we need CI/CD in ML?

> Continuous Delivery is the ability to get changes of all types -
> including new features, configuration changes, bug fixes, and experiments -
> into production, or into the hands of users, safely and quickly in a sustainable way.
>
> Jez Humble and Dave Farley, https://continuousdelivery.com/

> Continuous Delivery for Machine Learning (CD4ML)
> is a software engineering approach in which a cross-functional team
> produces machine learning applications based on code, data, and models
> in small and safe increments that can be reproduced and reliably released at any time,
> in short adaptation cycles.
>
> -- <cite>Martin Fowler, https://martinfowler.com/articles/cd4ml.html</cite>

In short, CI/CD approach in MLOps helps us develop, manage, and deliver ML-based
software in an automated way. Different teams, responsible e.g. for data, modeling and
infrastructure management, can develop their parts independently, relying on automated
processes for repeatable tasks. CI/CD ensures reproducibility throughout the entire cycle
and enables us to continuously ship new versions of the software into production.

![CI_CD](images/ci_cd_img.png)

Source - [DataCamp](https://www.datacamp.com/tutorial/ci-cd-for-machine-learning?dc_referrer=https%3A%2F%2Fwww.perplexity.ai%2F)

### GitHub Actions

It's a tool that allows us to build and run automated pipelines (workflows) using YAML definitions.
Public repositories get unlimited hours and quite powerful machines, enough for almost all
open source projects.

[Features of GitHub Actions](https://github.com/features/actions) include:

1. **Hosted runners**, e.g. Linux, macOS, Windows, run directly on VM or inside a container.
2. **Many languages**, e.g. Python, Java, Node.js.
3. **Logs** allowing monitoring in real-time and debugging failures.
4. **Environments variables & secrets store** built in and easy to use.

### Setting up a GitHub repository

Simply create new public repository on GitHub and setup connection to the repo locally.
Public repositories get unlimited hours for GitHub Actions workflows.
See [documentation](https://docs.github.com/en/actions/using-github-hosted-runners/using-github-hosted-runners/about-github-hosted-runners#standard-github-hosted-runners-for-public-repositories)
for details.

### Using GitHub Codespaces (optional)

[GitHub Codespaces](https://github.com/features/codespaces) is basically a virtual session in the browser
with VS Code ready to use. It contains preinstalled software like popular programming languages, Docker,
curl, and more. It is great as a learning platform and for quickly trying out things.

It is free for individual use up to 60 hours per month, with machines having 2 cores, 4 GB of RAM and
15 GB of storage. After that, it uses pay-as-you-go pricing, so don't exceed the time limit. For the
purpose of this lab, this will very easily suffice.

### Preparing our first GitHub Actions workflow

1. Create `.github/workflows` directory. All workflows are defined there.
2. Create file `hello_world.yaml` there.
3. Write workflow that:
    - runs on `ubuntu-latest` image
    - prints `Hello world!` using `echo` command

```yaml
name: Hello World workflow
on:
  # workflow_dispatch option enables manual trigger button
  workflow_dispatch:

# under job key we define the names and jobs definitions, typically as multiple steps
# e.g. what commands to run, what environment to set
jobs:
  # our job name: `hello-world-job`
  hello-world-job:
    # our job definition
    runs-on: ... # here, define image on which we will run our job

    # definitions of all job steps
    steps:
      - name: Print Hello World # step name, displayed on UI
        run: ... # print "Hello world!" here
```

4. Commit changes
5. Navigate to GitHub -> your repository -> Actions -> All workflows -> Hello World workflow
   ![actions](images/actions.png)
6. Click on `Run workflow` button and trigger job.
   ![workflow dispatch](images/workflow_dispatch.png)
7. Refresh the page and navigate to executed job. When you click on rectangle with step name,
   you will see list of steps and their statuses. You can also inspect logs printed by each step.
8. Document your result, e.g. by screenshot with `Hello World` printed.

---

### Exploring GitHub Actions features

**Workflows vs Jobs vs Steps**

**Workflows** are the highest-level configuration in GitHub Actions. They are
triggered by events like:

- commiting pushes to branches
- pull requests, e.g. on opening or closing
- on schedule, e.g. using cron scheduling

They contain one or more jobs, each performing different tasks, e.g. builds with
different OS configurations. In MLOps, we can define separate workflows for individual
tasks, like code testing or model deployment.

**Jobs** are groups of steps, each running on a given virtual machine. Jobs are isolated,
i.e. each one gets a fresh environment and can run independently of other jobs. By default,
jobs run in parallel for efficiency, but you can also set inter-job dependencies. This is
useful in MLOps when you first verify code in a lightweight VM, and then, if everything goes
well, larger VM is used for model compilation and deployment.

**Steps** are individual tasks within a job, e.g. commands or actions. They run sequentially,
in order, inside the same job environment. They can modify it, sharing files and environment
variables. In MLOps, those can include running code checkers, linters, or tests.

Summarizing:

- workflow contains jobs, job consists of steps
- workflows are triggered by events
- jobs run inside the workflow, independently (in parallel) or with defined dependencies
- steps are specific commands, executed sequentially, that do the actual work in jobs

### Workflow events

**Workflow events** in GitHub Actions are specific activities in your repository that can
automatically start (trigger) your workflows. Those are e.g. pushing commits, opening a PR,
or adding a Git tag. You define which events trigger a workflow, using the `on` key in
workflow definition. Some are more configurable, e.g. run only on comment creation in a PR,
not edit or delection.

Workflow events can be used in MLOps e.g. to run tests on each commit, or to deploy a new
model version upon tagging a Git commit.

```yaml
# on: Git commit push
name: My Workflow on push

on: push

jobs:
  ...
```

```yaml
# on: pull_request - with filtering by event_types / branches
name: My Workflow on pull_request with filtering

on:
  pull_request:
    types:
      - opened
      - synchronized
    branches:
      - master
      - develop

jobs:
  ...
```

### Contexts

**Contexts** are used to access dynamic or configured values in workflows.
Those can be e.g. repository or event information (`github`), environment
variables (`env`), user-provided inputs (`inputs`), or centrally managed
configuration (`vars`). Those are very indispensable for building configurable
workflows in MLOps, e.g. to select model version, access environment configuration,
or safely use secrets.

Let's take a look at a few contexts.

`github` [context](https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/accessing-contextual-information-about-workflow-runs#github-context)
is used to get information about repository, event, or workflow run itself.

```yaml
# github context
name: Github context example
on: workflow_dispatch
jobs:
  show-github-context:
    runs-on: ubuntu-latest
    steps:
      - name: Github Context
        run: | # "|" is a new line in YAML, useful for readable multiline commands
          echo "Event Name: ${{ github.event_name }}"
          echo "Ref: ${{ github.ref }}"
          echo "SHA: ${{ github.sha }}"
          echo "Actor: ${{ github.actor }}"
          echo "Workflow: ${{ github.workflow }}"
          echo "Run ID: ${{ github.run_id }}"
          echo "Run number: ${{ github.run_number }}"
```

`inputs` [context](https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/accessing-contextual-information-about-workflow-runs#inputs-context)
gets values provided by the user or another workflow as input configuration.

```yaml
# inputs context
name: Inputs context example
on:
  workflow_dispatch:
    inputs:
      to_deploy:
        type: boolean
        default: false
        description: 'Deploy app to production after integration and build step?'

jobs:
  show-input:
    runs-on: ubuntu-latest
    steps:
      - name: Inputs contexts
        run: |
          echo "Deploy?: ${{ inputs.to_deploy }}"
```

`env` [context](https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/accessing-contextual-information-about-workflow-runs#env-context)
is used to store and reuse configuration values across steps, jobs, or entire workflows. Note that
step-level environment overrides job- and workflow-level ones if necessary.

```yaml
# env context
name: Env context example
on: push

# workflow env (global for all jobs)
env:
  GREETING: Hello

jobs:
  show-env:
    runs-on: ubuntu-latest
    # job env (global for all steps within job)
    env:
      TARGET: World
    steps:
      - name: Env context
        run: |
          echo "${{ env.GREETING }} ${{ env.TARGET }}"
      - name: Overwritten env context
        env: # step env (local env for specific step)
          GREETING: HELLO!
        run: |
          echo "${{ env.GREETING }} ${{ env.TARGET }}"
```

`vars` [context](https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/accessing-contextual-information-about-workflow-runs#vars-context)
are used for non-sensitive environment variables configuration values
that you want to reuse, e.g. base URLs for services.

```yaml
# vars context
# variables are repository-level variables that can be reused in many individual workflows
name: Vars Context Example

on: workflow_dispatch

jobs:
  show-vars:
    runs-on: ubuntu-latest
    steps:
      - name: Show a configuration variable
        run: |
          echo "Config variable: ${{ vars.MY_CONFIG }}"
```

`secrets` [context](https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/accessing-contextual-information-about-workflow-runs#secrets-context)
is used for sensitive data that must be kept secure and not shown in logs, e.g.
API keys or access codes.

```yaml
# secrets context
# secrets are repository-level variables that should be encrypted
# and can be reused in many individual workflows

name: Secrets Context Example

on: workflow_dispatch

jobs:
  show-secrets:
    runs-on: ubuntu-latest
    steps:
      - name: Use a secret (masked in logs)
        run: |
          echo "API Key: ${{ secrets.API_KEY }}"
```

---

## 2. Project setup

In this section, we will enhance the project from previous introductory labs to MLOps
and cloud computing on AWS.

### Exercise 1 (1 point)

Prepare the environment and dependencies.

1. **Dependencies**: Edit `pyproject.toml` to organize dependencies into three distinct groups:
    - `integration`
    - `deployment`
    - `inference`

Ensure that in inference group you do not have heavy libraries like `transformers` or `torch`. We will use onnxruntime
and tokenizers instead.

2. **S3 Model Storage**:
    - Download the model artifacts
      from [Google Drive](https://drive.google.com/file/d/1NRZdYq5jweVRUzAZG518LMhs4E56IgxG/view).
    - Create a new S3 bucket (e.g., `mlops-lab11-models-<your-name>`) in `us-east-1`.
    - Upload the downloaded model files (`sentence_transforme.model` folder, `classifier.joblib`) to the bucket.

## 3. CI pipeline

We will create a GitHub Actions workflow to automate integration checks.

### Exercise 2 (2 points)

Create `.github/workflows/ci_cd_workflow.yaml` that triggers on `workflow_dispatch`.

1. Define an `integration` job running on `ubuntu-latest`.
2. Use the `actions/checkout@v4` action to clone the repository.
3. Install `uv` using the official installer.
4. Install dependencies for the `integration` group only (use `uv sync` with appropriate flags).
5. Add steps to run `ruff check` and `pytest tests`.
6. Add global environment variables (before `jobs:` section) for AWS region, ECR repository, and S3 bucket name.

**Workflow Skeleton:**

```yaml
name: CI/CD workflow

on: workflow_dispatch

env:
  AWS_REGION: # your region
  ECR_REPOSITORY: # your ecr repository name
  S3_BUCKET: # your s3 bucket name

jobs:
  integration:
    name: checks_and_tests
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code repo
        # This step checks out your repository code so the workflow can access it
        uses: actions/checkout@v4

      - name: Setup uv
        uses: astral-sh/setup-uv@v7

      - name: Install dependencies
        # TODO: Add command to sync ONLY integration group dependencies
        run: ...

      # TODO: Add steps for 'ruff check' and 'pytest'
```

## 4. Optimization & containerization

We want to deploy an optimized ONNX model. The flow will be:
**Input (text)** -> **Tokenizer** -> **ONNX model** (Embeddings) -> **Joblib classifier** -> **Prediction (output)**.

### Exercise 3 (3 points)

1. **Settings**:
    - Create `src/scripts/settings.py` with a `Settings` class to manage paths.
    - Add S3 bucket name and model paths.
    - Add local paths for saving artifacts, ONNX model, and tokenizer.

2. **Helper scripts** - prepare Python scripts in `src/scripts/`:
    - `download_artifacts.py`: Download model artifacts from your S3 bucket using `boto3`.
      Check [boto3 file download documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-example-download-file.html).
    - `export_classifier_to_onnx.py`: Convert the `classifier.joblib` to ONNX format using `skl2onnx`.
      Check [skl2onnx documentation](https://onnx.ai/sklearn-onnx/).
    - `export_sentence_transformer_to_onnx.py`:
      Convert the PyTorch model to ONNX format. You must also save the tokenizer files so they are available for
      inference.

```python
# export_classifier_to_onnx.py

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from settings import Settings


def export_classifier_to_onnx(settings: Settings):
    print(f"Loading classifier from {settings.classifier_joblib_path}...")
    classifier = joblib.load(settings.classifier_joblib_path)

    # define input shape: (batch_size, embedding_dim)
    initial_type = [("float_input", FloatTensorType([None, settings.embedding_dim]))]

    print("Converting to ONNX...")
    onnx_model = convert_sklearn()  # TODO: complete conversion here

    print(f"Saving ONNX model to {settings.onnx_classifier_path}...")
    # TODO: save the onnx_model to settings.onnx_classifier_path
```

```python
# export_sentence_transformer_to_onnx.py

import os
from settings import Settings
import torch
from transformers import AutoTokenizer, AutoModel


# Wrapper to include Mean Pooling in the ONNX graph
class SentenceEmbeddingModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Last hidden state: (Batch, Seq, Hidden)
        last_hidden_state = outputs.last_hidden_state

        # Mean Pooling operation
        # attention_mask: (Batch, Seq) -> expand to (Batch, Seq, Hidden)
        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )

        # Sum embeddings where mask is 1
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)

        # Sum mask to get count of valid tokens
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)

        # Divide to get mean
        mean_pooled = sum_embeddings / sum_mask
        return mean_pooled


def export_model_to_onnx(settings: Settings):
    # Use AutoModel to get embeddings (not classification logits)
    base_model = AutoModel.from_pretrained(settings.sentence_transformer_dir)
    tokenizer = AutoTokenizer.from_pretrained(settings.sentence_transformer_dir)

    # Wrap model
    model = SentenceEmbeddingModel(base_model)

    model.eval()
    dummy_text = "This is a sample input for ONNX export."
    inputs = tokenizer(dummy_text, return_tensors="pt")

    onnx_path = settings.onnx_embedding_model_path
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs.get("attention_mask")),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["sentence_embedding"],  # pooled embedding (Batch, Hidden)
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "sentence_embedding": {0: "batch_size"},
            },
            opset_version=18,
            dynamo=False,
        )

    tokenizer.save_pretrained(os.path.dirname(settings.onnx_tokenizer_path))

    print(f"ONNX model exported to {onnx_path}")
    return onnx_path

```

3. **Refactor application for inference**:
   Create `sentiment_app/app.py` this will be your FastAPI application using `onnxruntime`.

   **Optimization note**: Instead of installing the heavy `transformers` library, use the
   lightweight implementation in Rust, `tokenizers` library. You will need to add it to your
   `inference` dependencies.

    - Load the tokenizer
   ```python
   from tokenizers import Tokenizer

   tokenizer = Tokenizer.from_file("path/to/tokenizer.json")
   ```
    - Load the ONNX model using `onnxruntime.InferenceSession`
   ```python
   import onnxruntime as ort
   ort_session = ort.InferenceSession("path/to/onnx_model.onnx")
   ```
    - Load the ONNX classifier using `onnxruntime.InferenceSession`
   ```python
   ort_classifier = ort.InferenceSession("path/to/onnx_classifier.onnx")
   ```
    - Prepare inference and `/predict` endpoint.
   ```python
     # tokenize input
     encoded = self.tokenizer.encode(cleaned_text)

     # prepare numpy arrays for ONNX
     input_ids = np.array([encoded.ids])
     attention_mask = np.array([encoded.attention_mask])

     # run embedding inference
     embedding_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
     embeddings = self.embedding_session.run(None, embedding_inputs)[0]

     # run classifier inference
     classifier_input_name = self.classifier_session.get_inputs()[0].name
     classifier_inputs = {classifier_input_name: embeddings.astype(np.float32)}
     prediction = self.classifier_session.run(None, classifier_inputs)[0]

     label = SENTIMENT_MAP.get(prediction[0], "unknown") # return this label as response
   ```

4. **Optimized Dockerfile**: Create 2 Docker files.

- Use multi-stage builds to keep the final image size small.
- Use the `uv` base image for building to ensure fast and reliable dependency installation.
- Call them `Dockerfile` and `Dockerfile.dev` (for local testing with all dependencies).
- Specify `.dockerignore` to exclude unnecessary files from the build context.
- Both should look similar:

  ```Dockerfile
  # Dockerfile.dev
  # Build stage
  FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
  
  WORKDIR /app
  
  # Install dependencies
  COPY pyproject.toml uv.lock ./
  # Sync only inference group to .venv
  RUN uv sync --frozen --group inference --no-install-project
  
  # Runtime stage
  FROM python:3.12-slim-bookworm
  
  WORKDIR /app
  
  # Copy virtual environment
  COPY --from=builder /app/.venv /app/.venv
  ENV PATH="/app/.venv/bin:$PATH"

  # Copy application code
  COPY sentiment_app ./sentiment_app
  
  # Copy model artifacts (ONNX + tokenizer.json + classifier.joblib)
  COPY model ./model
  
  # Run the application
  CMD ["uvicorn", "sentiment_app.app:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

5. **Local testing**:
   Before pushing to the cloud, verify your Docker image locally.
    1. **Build**: `docker build -t sentiment-onnx .`
    2. **Run**: `docker run -p 8000:8000 sentiment-onnx`
    3. **Test**: Send a curl request to `localhost:8000/predict`.

## 5. Continuous deployment (AWS)

Finally, we will push the image to ECR and deploy to AWS Lambda.

### Exercise 4 (4 points)

1. **GitHub Secrets**: Configure AWS credentials in your repository.
    - Go to **Settings** -> **Secrets and variables** -> **Actions**.
    - Click **New repository secret**.
    - Add `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` (if using Academy).

2. **ECR repository**: Create an AWS ECR repository in AWS Console named `sentiment-app-onnx` or use AWS CLI:
   ```bash
   aws ecr create-repository --repository-name sentiment-app-onnx --region us-east-1
   ```

3. **Deployment job**: Extend `ci_cd_workflow.yaml` with a `deployment` job.
    - Set `needs: integration` to run deployment only after integration passes.
    - Use `actions/checkout@v4` to clone the repository.
    - Configure AWS credentials (use `aws-actions/configure-aws-credentials@v4`).
    - Install `uv` and add it to PATH (same steps as in integration job).
    - Install `deployment` dependencies.
    - Add steps to execute your `main.py` script twice:
        - download artifacts from S3
        - export model to ONNX
    - Add steps to login to ECR and push Docker image with tag as `github.sha`:
       ```yaml
       - name: Login to ECR
         id: login-ecr
         uses: aws-actions/amazon-ecr-login@v2
         with:
           mask-password: 'true'

       - name: Build and Push Docker image
         env:
           REGISTRY: ${{ steps.login-ecr.outputs.registry }}
           REPOSITORY: sentiment-app-onnx
           IMAGE_TAG: ${{ github.sha }}
         run: |
           docker build -t $REPOSITORY:$IMAGE_TAG .
           docker tag $REPOSITORY:$IMAGE_TAG $REGISTRY/$REPOSITORY:$IMAGE_TAG
           docker push $REGISTRY/$REPOSITORY:$IMAGE_TAG
       ```
4. **AWS Lambda compatibility**: Adjust your application to be compatible with Lambda.
   Frameworks like `mangum` allow us to use the same code for local ASGI server set up
   by FastAPI, as well as to deploy it to AWS Lambda and automatically handle Lambda
   function inputs and events.

- Add update pyproject.toml `inference` group with libraries: `mangum`, `awslambdaric`.
- Modify `app.py` to use `Mangum` adapter:
  ```python
  from mangum import Mangum

  app = FastAPI()
  # ... your existing code ...

  handler = Mangum(app)
  ```
- Create new final `Dockerfile` as a copy of `Dockerfile.dev`, but modify the entrypoint and command for Lambda:
  ```Dockerfile
  # Dockerfile
  # ... previous stages ...
  # comment out previous CMD
  # CMD ["uvicorn", "sentiment_app.app:app", "--host", "0.0.0.0", "--port", "8000"] 
  ENTRYPOINT ["python", "-m", "awslambdaric"]
  CMD ["app.handler"]
  ```

5. **SAM deployment**: [AWS SAM (Serverless Application Model)](https://github.com/aws/aws-sam-cli)
   enables easy deployment of serverless-based AWS services. It's less powerful and complex than
   Terraform and Pulumi, making it quite useful for deploying simple, separated services like single
   Lambda functions.

    - Create `sam-template.yaml` for a Lambda function.
   ```yaml
   AWSTemplateFormatVersion: '2010-09-09'
   Transform: AWS::Serverless-2016-10-31

   Parameters:
     ImageUri:
       Type: String
     LambdaExecutionRoleArn:
       Type: String

   Resources:
     SentimentFunction:
       Type: AWS::Serverless::Function
       Properties:
         PackageType: Image
         ImageUri: !Ref ImageUri
         Role: !Ref LambdaExecutionRoleArn
         MemorySize: 512
         Timeout: 30
         Events:
           Api:
             Type: HttpApi
             Properties:
               Path: /predict
               Method: post

   Outputs:
     ApiUrl:
       Value: !Sub "https://${ServerlessHttpApi}.execute-api.${AWS::Region}.amazonaws.com/predict"
   ```
    - In the `deployment` job, add `Delete stack if in rollback state` step:
   ```yaml
   - name: Delete stack if in rollback state
     run: |
       STATUS=$(aws cloudformation describe-stacks \
         --stack-name sentiment-app-stack \
         --region ${{ env.AWS_REGION }} \
         --query 'Stacks[0].StackStatus' \
         --output text 2>/dev/null || echo "NOT_FOUND")

       echo "Current stack status: $STATUS"

       case "$STATUS" in
         *ROLLBACK* )
           aws cloudformation delete-stack --stack-name sentiment-app-stack --region ${{ env.AWS_REGION }}
           aws cloudformation wait stack-delete-complete --stack-name sentiment-app-stack --region ${{ env.AWS_REGION }}
           ;;
       esac
   ```

    - Add `Resolve Lambda role ARN from name` step:
   ```yaml
   - name: Resolve Lambda role ARN from name
     id: lambda_role
     run: |
       ROLE_NAME="LabRole"
       ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)
       echo "Resolved role ARN: $ROLE_ARN"
       echo "role_arn=$ROLE_ARN" >> $GITHUB_OUTPUT
   ``` 

    - Add `Deploy to AWS Lambda using SAM` step:
   ```yaml
   - name: Deploy with AWS SAM
     env:
       REGISTRY: ${{ steps.login-ecr.outputs.registry }}
       REPOSITORY: ${{ env.ECR_REPOSITORY }}
       IMAGE_TAG: ${{ github.sha }}
     run: |
       sam deploy \
         --template-file sam-template.yaml \
         --stack-name sentiment-app-stack \
         --image-repository $REGISTRY/$REPOSITORY \
         --parameter-overrides \
           ImageUri=$REGISTRY/$REPOSITORY:$IMAGE_TAG \
           LambdaExecutionRoleArn=${{ steps.lambda_role.outputs.role_arn }} \
         --capabilities CAPABILITY_IAM \
         --no-confirm-changeset \
         --no-fail-on-empty-changeset \
         --region ${{ env.AWS_REGION }}
   ```

### Verification

1. Check the API Gateway URL in the logs.
2. Send a test request:
   ```bash
   curl -X POST <YOUR_API_URL>/predict -H "Content-Type: application/json" -d '{"text": "MLOps is amazing!"}'
   ```

### Grading [10 points]

1. Project Setup & S3 (Exercise 1) [1 point]
2. CI pipeline (Exercise 2) [2 points]
3. Optimization scripts & containerization (Exercise 3) [3 points]
4. Continuous deployment to AWS (Exercise 4) [4 points]
