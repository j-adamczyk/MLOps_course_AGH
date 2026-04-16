# Laboratory 10 - Pulumi

## Introduction

Pulumi is an Infrastructure as Code tool that lets you define cloud resources using Python
(or TypeScript, Go) instead of a domain-specific configuration language. You write a regular
Python program - with loops, functions, classes, conditionals - and Pulumi executes it to
provision infrastructure. Pulumi manages state and plans changes declaratively, so the
development loop feels similar to Terraform: preview what will change, apply, destroy.

A Pulumi project has three key files:
- `Pulumi.yaml` - project metadata and runtime configuration
- `__main__.py` - your infrastructure code, the entry point
- `Pulumi.<stack>.yaml` - per-stack configuration (generated, can contain secrets)

**Stack** is Pulumi's concept for an isolated deployment of the same infrastructure, e.g.
`dev`, `staging`, `prod`. Each stack has its own state and configuration.

To pass, submit:
- a link to a public GitHub repository with your Pulumi code
- a report (PDF or DOCX) with screenshots showing the created resources in AWS/GitHub UI and
  `pulumi up` logs confirming the resources were provisioned by Pulumi, not created manually


## Setting up

### Prerequisites

1. Make sure [Pulumi CLI](https://www.pulumi.com/docs/install/) is installed. After installation verify via running below command in the terminal:
```bash
pulumi version
```

2. Make sure [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
   is configured.
-  Install the AWS CLI by following the [AWS CLI installation guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and verify the
installation with `aws --version`.
- `(PERSONAL ACCOUNT)` In the IAM Console, select your new user and navigate to the **Security
Credentials** tab.
- `(PERSONAL ACCOUNT)` Create a new access key and store it securely (treat it like a password—do not
share it!.
- `(AWS ACADEMY)` When you start your **awsadacemy** session under `AWS Details` the access keys
will be displayed. Copy them and also the session token and paste to `~/.aws/credentials`. When using `aws configure` it will ask you only for access key and secret key, but you also need to add the session token manually. The file should look like this:
```ini
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
aws_session_token = YOUR_SESSION_TOKEN # not needed if you are using you private account
region = YOUR_REGION # e.g., us-east-2 or eu-west-1
```
- verify your connection via:

```bash
aws sts get-caller-identity
```

3. Generate a GitHub Personal Access Token (PAT). Make sure you **never publish it** and do not
   put it in your repository. To create a token:
   1. Go to GitHub - Settings - Developer settings - Personal access tokens - Tokens (classic).
   2. Click "Generate new token" - "Generate new token (classic)".
   3. Give your token a descriptive name.
   4. Select the necessary scopes: `repo`, `admin:repo_hook`, `read:org`, `delete_repo`. After finishing this lab drop this token :) 
   5. Click "Generate token" and copy the token immediately. It will not be shown again.

### uv project setup

```bash
uv sync
source .venv/bin/activate
```

The `pyproject.toml` already declares all dependencies. `uv sync` installs them into `.venv`.

### Pulumi project files

The `Pulumi.yaml` in this directory is configured to use uv as the Python toolchain:

```yaml
name: pulumi-lab
runtime:
  name: python
  options:
    toolchain: uv
description: MLOps Lab 10 - Pulumi
```

### State backend

Pulumi stores state (what infrastructure exists) either in Pulumi Cloud or a backend of your
choice. For this lab, use local state to keep things simple:

```bash
export PULUMI_CONFIG_PASSPHRASE=""
pulumi login --local
```

`PULUMI_CONFIG_PASSPHRASE` encrypts secrets in the local state file. Empty string is fine for a
lab; in production you would set a real passphrase or use Pulumi Cloud / S3 backend. See
[state and backends](https://www.pulumi.com/docs/concepts/state/) and
[secrets management](https://www.pulumi.com/docs/concepts/secrets/) in the Pulumi docs.

### Stack and secrets

```bash
pulumi stack init dev
pulumi config set --secret github_token YOUR_GITHUB_TOKEN
```

`pulumi config set --secret` encrypts the value before storing it in `Pulumi.dev.yaml`. Never
commit actual secret values - the file stores only the encrypted ciphertext, which is safe to
commit.

Commit:
- `__main__.py` and any other `.py` files
- `Pulumi.yaml`
- `Pulumi.dev.yaml` - contains encrypted secrets, safe to commit
- `pyproject.toml` and `uv.lock`

Do not commit:
- `.pulumi/` - local state directory, machine-specific
- `.venv/`
- `.env` if you use one

## Pulumi CLI cheatsheet

```bash
pulumi preview
```
Shows what Pulumi will create, update, or delete without making any changes.

```bash
pulumi up
```
Runs preview and prompts for confirmation, then applies changes.

```bash
pulumi destroy
```
Destroys all resources in the current stack.

```bash
pulumi stack output
```
Prints all exported outputs of the current stack.

```bash
pulumi stack ls
```
Lists all stacks in the current project.

```bash
pulumi config
```
Lists all configuration values for the current stack.

```bash
pulumi logs
```
Retrieves logs from running resources (e.g. Lambda).

## GitHub repository setup

Let's create a GitHub repository and configure branch protection for `main`. This example
introduces two core Pulumi concepts: providers and resource dependencies.

Create a new file `github.py` and a `__main__.py` that imports it, or put everything in
`__main__.py` for now.

```python
# __main__.py

import pulumi
import pulumi_github as github

config = pulumi.Config()
github_token = config.require_secret("github_token")
# change to "public" if you want the repository to be publicly visible
repo_visibility = config.get("repo_visibility") or "private"

provider = github.Provider("github-provider", token=github_token)

repo = github.Repository("lab-repo",
    name="pulumi-managed-repo",
    description="Repository managed by Pulumi",
    visibility=repo_visibility,
    auto_init=True,
    opts=pulumi.ResourceOptions(provider=provider)
)

branch_protection = github.BranchProtection("main-protection",
    repository_id=repo.node_id,
    pattern="main",
    enforce_admins=True,
    opts=pulumi.ResourceOptions(provider=provider, depends_on=[repo])
)

pulumi.export("repo_url", repo.html_url)
pulumi.export("branch_protection_id", branch_protection.id)
```

Note `depends_on=[repo]` in `BranchProtection` - Pulumi resolves the dependency graph
automatically when you reference a resource's output (e.g. `repo.node_id`), but `depends_on`
makes the ordering explicit when there is no direct output reference.

Run `pulumi preview` to see the plan, then `pulumi up` to apply. Verify the repository and branch
protection exist on GitHub before moving on.

### Exercise 1 - GitHub repository

Extend the configuration above:

1. Add `required_pull_request_reviews` to the branch protection rule, requiring at least 1
   approving review before merging. See the
   [BranchProtection docs](https://www.pulumi.com/registry/packages/github/api-docs/branchprotection/).
2. The walkthrough already reads `repo_visibility` from Pulumi config with a `"private"` default.
   Set it to `"public"` for your deployment using:
   ```bash
   pulumi config set repo_visibility public
   ```
   Config values are stored in `Pulumi.dev.yaml` and read at deploy time via `config.get()`.
   See [Pulumi config docs](https://www.pulumi.com/docs/concepts/config/) for more details.
3. After `pulumi up`, verify the repository and branch protection exist on GitHub.
4. **Document your deployment**: save the `pulumi up` output as proof.
5. Run `pulumi destroy` to clean up.

## S3 buckets and assets

### Uploading a file

The example below creates an S3 bucket and uploads a text file to it. `content` accepts an
inline string; `source` accepts a local file path.

```python
# __main__.py

import pulumi
import pulumi_aws as aws

bucket = aws.s3.Bucket("lab-bucket",
    tags={"Name": "pulumi-lab"}
)

uploaded_file = aws.s3.BucketObject("hello-file",
    bucket=bucket.id,
    key="hello.txt",
    content="Hello from Pulumi!",
    content_type="text/plain"
)

pulumi.export("bucket_name", bucket.id)
pulumi.export("bucket_arn", bucket.arn)
```

Run `pulumi up`, then verify the file exists:
```bash
aws s3 ls s3://$(pulumi stack output bucket_name)
```

### Exercise 2 - Static website

Extend the above to serve a static website. You need four resources in total. Add them one by
one and run `pulumi preview` after each to see what will change.

1. Enable website hosting on the bucket using
   [`BucketWebsiteConfiguration`](https://www.pulumi.com/registry/packages/aws/api-docs/s3/bucketwebsiteconfiguration/).
   Set `index_document` to `index.html`:
   ```python
   website = aws.s3.BucketWebsiteConfiguration("website",
       bucket=bucket.id,
       index_document=aws.s3.BucketWebsiteConfigurationIndexDocumentArgs(
           suffix="index.html"
       )
   )
   ```

2. Disable the default S3 public access block using
   [`BucketPublicAccessBlock`](https://www.pulumi.com/registry/packages/aws/api-docs/s3/bucketpublicaccessblock/).
   All four `block_*` settings must be set to `False`, otherwise the bucket policy in the next
   step will be rejected by AWS. The argument names are listed in the docs under "Inputs".


   ```python
    pab = aws.s3.BucketPublicAccessBlock("public-access-block",
        bucket=bucket.id,
        ...
    )
   ```

4. Attach a bucket policy that allows public reads. The policy ARN must be built from
   `bucket.arn`, which is a `pulumi.Output` - you cannot concatenate it with a plain string.
   Use [`pulumi.Output.format()`](https://www.pulumi.com/docs/concepts/inputs-outputs/#outputs-and-strings):
   ```python
   policy_document = pulumi.Output.format('{{"Version":"2012-10-17","Statement":[{{"Effect":"Allow","Principal":"*","Action":"s3:GetObject","Resource":"{0}/*"}}]}}', bucket.arn)

   aws.s3.BucketPolicy("bucket-policy", 
        bucket=bucket.id,
        policy=policy_document, 
        opts=pulumi.ResourceOptions(depends_on=[pab])
    )
   ```

5. Upload `index.html` as a [`BucketObject`](https://www.pulumi.com/registry/packages/aws/api-docs/s3/bucketobject/)
   with `content_type="text/html"` and any HTML content passed as `content`:
   ```python
   aws.s3.BucketObject("index-html",
       bucket=bucket.id,
       key="index.html",
       content="<h1>Hello from Pulumi!</h1>",
       content_type="text/html"
   )
   ```

6. Export the website URL:
   ```python
   pulumi.export("website_url", website.website_endpoint)
   ```

7. Run `pulumi up`, then open the exported URL in a browser to verify.
8. **Document your deployment**: screenshot the website URL and save the `pulumi up` output.
9. Run `pulumi destroy`.

## Multi-region buckets with a loop

In Terraform, deploying to multiple regions requires defining one provider alias per region and
one resource block per region. In Python, you use a loop.

```python
# __main__.py

import pulumi
import pulumi_aws as aws

regions = ["us-east-1", "us-west-2"]
buckets = []

for region in regions:
    provider = aws.Provider(f"provider-{region}", region=region)

    bucket = aws.s3.Bucket(f"bucket-{region}",
        tags={"Region": region},
        opts=pulumi.ResourceOptions(provider=provider)
    )

    aws.s3.BucketVersioning(f"versioning-{region}",
        bucket=bucket.id,
        versioning_configuration=aws.s3.BucketVersioningVersioningConfigurationArgs(
            status="Enabled"
        ),
        opts=pulumi.ResourceOptions(provider=provider)
    )

    buckets.append(bucket)

pulumi.export("bucket_names", [b.id for b in buckets])
pulumi.export("bucket_arns", [b.arn for b in buckets])
```

### Exercise 3 - Multi-region buckets

1. Add a third region of your choice.
2. Add a lifecycle rule on each bucket transitioning objects to `GLACIER` after 90 days.
   See [`BucketLifecycleConfiguration`](https://www.pulumi.com/registry/packages/aws/api-docs/s3/bucketlifecycleconfiguration/)
   and [`BucketVersioning`](https://www.pulumi.com/registry/packages/aws/api-docs/s3/bucketversioning/) already in the walkthrough as a structural reference.
3. Export a dict mapping region name to bucket ARN.
4. **Document your deployment**: save the `pulumi up` output.
5. Run `pulumi destroy`.

## Reusable components with ComponentResource

Pulumi's equivalent of a Terraform module is a `ComponentResource` - a Python class that groups
related resources. The class manages its own child resources and exposes outputs as attributes.

```python
# components.py

import pulumi
import pulumi_aws as aws

class RegionalBucket(pulumi.ComponentResource):
    def __init__(self, name: str, region: str, lifecycle_days: int = 90, opts=None):
        super().__init__("lab:index:RegionalBucket", name, {}, opts)

        child_opts = pulumi.ResourceOptions(parent=self)

        provider = aws.Provider(f"{name}-provider",
            region=region,
            opts=child_opts
        )

        resource_opts = pulumi.ResourceOptions(parent=self, provider=provider)

        self.bucket = aws.s3.Bucket(f"{name}-bucket",
            tags={"Region": region},
            opts=resource_opts
        )

        aws.s3.BucketVersioning(f"{name}-versioning",
            bucket=self.bucket.id,
            versioning_configuration=aws.s3.BucketVersioningVersioningConfigurationArgs(
                status="Enabled"
            ),
            opts=resource_opts
        )

        aws.s3.BucketLifecycleConfiguration(f"{name}-lifecycle",
            bucket=self.bucket.id,
            rules=[aws.s3.BucketLifecycleConfigurationRuleArgs(
                id="glacier-transition",
                status="Enabled",
                transitions=[aws.s3.BucketLifecycleConfigurationRuleTransitionArgs(
                    days=lifecycle_days,
                    storage_class="GLACIER"
                )]
            )],
            opts=resource_opts
        )

        self.register_outputs({
            "bucket_id": self.bucket.id,
            "bucket_arn": self.bucket.arn
        })
```

```python
# __main__.py

import pulumi
from components import RegionalBucket

regions = ["us-east-1", "us-west-2"]
buckets = [RegionalBucket(f"lab-{r}", region=r) for r in regions]

pulumi.export("bucket_arns", {r: b.bucket.arn for r, b in zip(regions, buckets)})
```

### Exercise 4 - ComponentResource refactor

1. Copy your Exercise 3 code and refactor it to use `RegionalBucket`.
2. Add a `bucket_name_prefix` parameter to `RegionalBucket` that prepends a string to the
   bucket name.
3. Pass different `lifecycle_days` values to two of the buckets.
4. Verify the refactored code produces the same infrastructure as Exercise 3 (`pulumi preview`
   should show no changes if you migrate the state, or fresh creates if starting clean).
5. **Document your deployment**: save the `pulumi up` output.
6. Run `pulumi destroy`.

## Lambda with inline handler

This is where Pulumi's Python-native approach has a genuine advantage over Terraform. You can
write a Lambda handler function and deploy it in the same Python file, using
`inspect.getsource()` to extract the function source at deploy time.

```python
# __main__.py

import inspect
import pulumi
import pulumi_aws as aws

# Lambda handler - defined here, deployed below
def handler(event, context):
    import json
    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Hello from Pulumi Lambda!"})
    }

# IAM role for Lambda
role = aws.iam.Role("lambda-role",
    assume_role_policy=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    })
)

aws.iam.RolePolicyAttachment("lambda-basic-execution",
    role=role.name,
    policy_arn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
)

fn = aws.lambda_.Function("lab-function",
    code=pulumi.AssetArchive({
        "index.py": pulumi.StringAsset(inspect.getsource(handler))
    }),
    runtime=aws.lambda_.Runtime.PYTHON3D11,
    handler="index.handler",
    role=role.arn
)

pulumi.export("function_name", fn.name)
pulumi.export("function_arn", fn.arn)
```

After `pulumi up`, invoke the function to verify:
```bash
aws lambda invoke \
    --function-name $(pulumi stack output function_name) \
    --payload '{}' \
    response.json && cat response.json
```

### Exercise 5 - Lambda

1. Modify the handler to accept a `name` parameter from the event body and return
   `"Hello, {name}!"`. If `name` is not in the event, fall back to `"World"`.
2. Add a `LOG_LEVEL` environment variable to the Lambda function configuration.
3. Read `LOG_LEVEL` in the handler and print a log line using Python's `logging` module.
4. Invoke the function with `{"name": "MLOps"}` and verify the response.
5. Check CloudWatch logs: `pulumi logs`.
6. **Document your deployment**: save the invocation output and `pulumi up` output.
7. Run `pulumi destroy`.

## Grading [15 points] + [3 points extra]

1. GitHub repository + branch protection [3 points]
2. S3 static website [3.5 points]
3. Multi-region buckets with loop [4.5 points]
4. ComponentResource refactor [5 points]
5. Lambda with inline handler [3 points] - optional, but you are supposed to trigger lambda from the lab before exercise 5 and document this action via screenshot of response.