# Laboratory 10 - Terraform

## Introduction

This lab covers Infrastructure as Code using Terraform. If you prefer Python over HCL, see the
[Pulumi version](LAB_INSTRUCTION_PULUMI.md). For a general IaC overview and tool comparison, see
[LAB_INSTRUCTION.md](LAB_INSTRUCTION.md).

If you are using PyCharm, the [Terraform and HCL plugin](https://plugins.jetbrains.com/plugin/7808-terraform-and-hcl)
is recommended.

To pass, submit:
- a link to a public GitHub repository with your Terraform code
- a report (PDF or DOCX) with screenshots showing the created resources in AWS/GitHub UI and
  `terraform apply` logs confirming the resources were provisioned by Terraform, not created manually

## Why Terraform?

Terraform is cloud-agnostic and supports AWS, GCP, Azure, GitHub, and many more providers through
a plugin system. Its HCL language is declarative - you define what infrastructure you want, not
how to create it. Modules enable reuse across environments. Terraform tracks state, so it knows
what exists and what needs to change.

## Setting up Terraform

### Initial setup

Before you begin working with Terraform, ensure you have the following prerequisites in place:

1. Make sure that [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)
   is installed. Try running `terraform version` in your terminal.

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


3. Test that AWS CLI is working:
- List your S3 buckets: `aws s3 ls`. If configured correctly, you should see a list of your S3 buckets,
      or no output if you don't have any buckets.
- List your credentials: `aws sts get-caller-identity`. This will display your AWS account ID,
      confirming your credentials are working

4. Create a `.gitignore` file in your project root to avoid accidentally committing sensitive files or
   local Terraform state:
```gitignore
# .gitignore
*.tfvars
*.tfvars.json
**/.terraform/
terraform.tfstate
terraform.tfstate.backup
.terraform.tfstate.lock.info
```

Commit:
- `*.tf` files - your infrastructure definitions
- `.terraform.lock.hcl` - pins exact provider versions, ensures everyone on the team uses the same providers

Do not commit:
- `*.tfvars` / `*.tfvars.json` - contain secrets; pass sensitive values via environment variables instead
- `.terraform/` - downloaded provider binaries, regenerated automatically by `terraform init`
- `terraform.tfstate` / `terraform.tfstate.backup` - contain sensitive resource details in plaintext; use a remote backend

5. Generate GitHub Personal Access Token (PAT) for Terraform. Make sure you **never publish it**, and you
   don't put it in your repository! If you write it to file, make sure it's added to `.gitignore`. To create
   a token:
   1. Go to GitHub -> Settings -> Developer settings -> Personal access tokens -> Tokens (classic).
   2. Click "Generate new token" -> "Generate new token (classic)".
   3. Give your token a descriptive name.
   4. Select the necessary scopes: `repo`, `admin:repo_hook`, `read:org`.
   5. Click "Generate token" and copy the token immediately. It won't be shown ever again, and if you lose it.
      you will need to generate another one.

### Terraform CLI cheatsheet

```commandline
terraform init
```
Initializes a Terraform working directory, downloads providers and modules specified in the configuration.

```commandline
terraform plan
```
Creates an execution plan, showing what actions Terraform will take to change infrastructure to match
the configuration.

```commandline
terraform apply
```
Applies the changes required to reach the desired state of the configuration, creating or updating infrastructure.

```commandline
terraform destroy
```
Destroys all resources managed by the current Terraform configuration, removing infrastructure.

```commandline
terraform validate
```
Validates the syntax and configuration files for errors, without accessing any remote services.

```commandline
terraform fmt
```
Rewrites configuration files to a canonical format and style for consistency.

```commandline
terraform state
```
Advanced command for manipulating the state file, with subcommands like `list`, `show`, `mv`, `rm`.

```commandline
terraform console
```
Interactive console for evaluating expressions and testing interpolations.

```commandline
terraform providers
```
Shows information about providers used in the configuration.

## GitHub repository IaC setup

Let's create a simple Terraform project that provisions a GitHub repository.

1. Create project directory.
2. Create file `variables.tf`. This file defines input variables for your Terraform configuration, allowing
   you to parameterize your infrastructure. Using variables instead of hardcoding values directly in resource
   declarations makes your configuration more flexible, reusable, and secure for sensitive information.
```hcl
# variables.tf

variable "github_token" {
  description = "GitHub Personal Access Token"
  type        = string
  sensitive   = true  # Marks this variable as sensitive, preventing it from appearing in logs and console output
}

variable "repository_name" {
  description = "Name of the GitHub repository to create"
  type        = string
  default     = "terraform-managed-repo"
}

variable "repository_description" {
  description = "Description of the GitHub repository"
  type        = string
  default     = "Repository managed by Terraform"
}
```
3. Create file `main.tf`. This is the primary configuration file, where you define your infrastructure resources.
   It contains the provider configuration, resource definitions, and outputs.
```hcl
# main.tf

terraform {
  required_providers {
    github = {
      source  = "integrations/github"
      version = "~> 5.0"
    }
  }
}
```

In above code we defined required providers, in particular defining GitHub as a provider.
Each Terraform module must declare its [required providers](https://developer.hashicorp.com/terraform/language/providers/requirements),
so that Terraform can install and use them. Providers are plugins that Terraform uses to
create and manage your resources, making the infrastructure modular and enabling multi-cloud
management in terraform.

Code elements:
- provider requirements are declared in a `required_providers` block
- provider requirement consists of a local name, a source location, and a version constraint
- [provider source](https://developer.hashicorp.com/terraform/language/providers/requirements#source-addresses)
  tells Terraform where to find the provider
- version constraint - `~> 5.0` means "any version that is compatible with 5.0", i.e. `5.0.x` but not `6.x`

4. Add GitHub token variable:
```hcl
# main.tf

provider "github" {
  token = var.github_token
}
```

This code [configures the provider](https://developer.hashicorp.com/terraform/language/providers/configuration),
allowing Terraform to interact with GitHub API, authenticating with it using
[the provided token][GitHub Provider Documentation](https://registry.terraform.io/providers/integrations/github/latest/docs#token).

Note the use of `var.` - this means that you access variable, defined in `variables.tf` file. This allows
us to keep sensitive information out of our main configuration. 

5. Add our first resource - GitHub repository:
```hcl
# main.tf
# above code

resource "github_repository" "example" {
  name        = var.repository_name
  description = var.repository_description
  visibility  = "private"
  auto_init   = true
}
```

This declares that we want to have a repository in our infrastructure. Resources are infrastucture
objects, configurable with options depending on a particular resource type, e.g. `name` or `visibility`
in case of GitHub repositories. See [GitHub Repository Resource Documentation](https://registry.terraform.io/providers/integrations/github/latest/docs/resources/repository)
for more details.

6. Define module output:
```hcl
# main.tf
# above code

output "repository_url" {
  value       = github_repository.example.html_url
  description = "URL of the created repository"
}
```

Outputs are similar to variables, exported from the current module. They are often generated
by providers, e.g. here the repository URL will be generated upon resource creation. Note that
in `github_repository.` we reference the resource, but we don't have to control the exact API
calls or order of operations. This is due to the declarative nature of Terraform.

Outputs allow you to further use the dynamically created values, e.g. display them or pass
them to other parts of code.

7. Export GitHub token as an environment variable:
```bash
export TF_VAR_github_token=github_token_here
```

8. Run `terraform init` to install providers and set up the module.
9. Running `terraform validate` checks if your configuration is correct, and `terraform fmt`
   reformats your code to keep a uniform standard. Using them is a good practice.
10. Check the changes plan with `terraform plan`. Read the detailed plan that Terraform
    will use to create your resources. Analyzing this is crucial, particularly to check
    any destructive actions.
11. Run `terraform apply` and observe the output. Confirm the changes by typing `yes`.
    Validate if your repository has been created.
12. Finally, clean everything with `terraform destroy`.

### Exercise 1

In this exercise, you will refactor the GitHub repository configuration. Previous code was good
as a proof-of-concept (PoC), but keeping everything in a single `main.tf` is definitely not
scalable.

1. Create new `.tf` file with readable name, describing infrastructure inside.
2. Move the repository configuration into this file.
3. Introduce new variable, which will manage the visibility (private / public) of our repository:
   - boolean type
   - readable name, e.g. `publicly_visible`
   - default value `false`
4. At the top of your `.tf` file, add a [local value](https://developer.hashicorp.com/terraform/language/values/locals).
   Locals allow you to perform inplace operations and transform inputs (variables).
```hcl
# your_file.tf

locals {
   visibility = var.<the variable name that you just created> ? "public" : 
   "private"
}
```
5. Use this local value in `github_repository` resource. Note that when using local values, use `local`:
```hcl
local.visibility  # NOT locals.
```

6. Move output to new file `outputs.tf`
```hcl
# outputs.tf

output "repository_url" {
  value       = ...
  description = ...
}
```

## State management and remote backends

Your infrastructure always has some current state. Managing this state allows Terraform to keep track
of what's present, and what needs to be changed depending on your declarations. There are 3 options for
this:
1. **Local**: Terraform state files are stored locally in project directory.
2. **Terraform Cloud**: Terraform state files are stored in Terraform Cloud.
3. **Terraform third party remote backends**: store state files on remote storage, e.g. S3, GCS, Azure Storage.

We commonly use **remote state backends**, as it has considerable advantages:
1. **Centralized management** - it's easier to track changes, collaborate with other team members,
   and enforce organization policies.
2. **Version control** - this works similarly to a remote code repository, allowing you to track
   and revert changes to the infrastructure state.
3. **Scalability** - for larger infrastructure deployments, it's easier to manage complex environments
   and multi-cloud setups this way.
4. **Auditability** - having a single remote state provides a clear audit trail of infrastructure changes,
   making it easier to identify and investigate any security incidents or misconfigurations.

### Configuring remote state

Here, we will use AWS S3. We have two options:
1. Create an AWS S3 bucket for the remote backend manually. The name should be meaningful and should indicate that it stores your Terraform state.
2. Create a new directory (Terraform project).
3. Create new file `providers.tf`:
```hcl
# providers.tf

terraform {
  required_version = ">=1.7.0"
  # Note that previously we did not declare terraform version. 
  # In this scenario terraform will not validate terraform version 
  # but try to resolve with installed one.
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # configure remote backend
  backend "s3" {
   region = "us-east-1"
   bucket = "your previously created bucket" 
   key    =  "your_project_name/terraform.tfstate"
   # if we would like to support locking, we need to provide a dynamodb table 
   # name, this might be helpful if we have multiple teams working on the same 
   # infrastructure
   # this is a good practice to avoid race conditions. The terraform will look if 
   # the lock is already taken and if it is, it will return an error until the 
   # lock is released.
   # dynamodb_table = "terraform-lock" 
   # you can omit this one for now.
   }
}

provider "aws" {
   region = "us-east-1"
}

```
4. Create `main.tf` and add any resource, e.g. [ECR](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/ecr_repository).

5. Run `terraform init` and `terraform apply`. Observe that Terraform will ask
   you to move your state to remote backend. Confirm it by typing `yes`.
6. Go to your S3 bucket and validate that `.tfstate` has been uploaded.
   Your remote state backend is now fully functional.
7. **Before destroying**, take a screenshot or save the `terraform apply` output and the S3 bucket
   contents as proof of successful deployment.
8. Run `terraform destroy` to clean up all created resources and avoid unnecessary AWS costs.

`.terraform.lock.hcl` file is generated by Terraform to manage dependencies and provide a consistent
version lock for your project. It **should be committed** into your repository to ensure that everyone
in team uses the same versions of providers and modules. If you're using Terraform Cloud, you don't need
to manually upload it to the repository though, it will automatically handle the versioning and sharing
of this for you.

### Aliases
Sometimes you may need to use the same provider, but with different configurations, e.g. for deploying
in different regions. [Aliases](https://developer.hashicorp.com/terraform/language/providers/configuration#alias-multiple-provider-configurations)
allow you to do exactly this, i.e. define multiple configurations for the same provider, and select which
one to use on a per-resource or per-module basis.

1. Create new directory (Terraform project).
2. Create file `providers.tf`:
```hcl
# providers.tf

terraform {
  # Note here that required_version is declared differently.
  # It means that the required version of Terraform is >=1.0 and <2.0.
  required_version = "~> 1.0" 
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# default provider configuration
provider "aws" {
  region = "us-east-1"
}

# provider alias with another region
provider "aws" {
  alias  = "us_west_2"
  region = "us-west-2"
}
```

```hcl
# s3.tf

# uses default provider configuration
resource "aws_s3_bucket" "my_bucket" {
  bucket = "my-bucket-afdab234423" # replace with your own unique name
  tags = {
    Name = "my-bucket"
  }
}

# alias provider specified, it will use its configuration
resource "aws_s3_bucket" "my_bucket_us_west_2" {
  bucket   = "my-bucket-afdab23442432432" # replace with your own unique name
  provider = aws.us_west_2
  tags = {
    Name = "my-bucket"
  }
}
```

3. Run `terraform init` and `terraform apply` to create the buckets.
4. **Document your deployment**: save the `terraform apply` output and/or a screenshot of the created
   S3 buckets in the AWS Console as proof of successful deployment.
5. Run `terraform destroy` to remove all created resources and avoid unnecessary AWS costs.

### Exercise 2

In this exercise, you'll deploy S3 buckets across multiple AWS regions. Multi-region setups are
common in practice for latency optimization, data residency compliance (e.g. GDPR), and disaster
recovery.

Exercise steps:
1. Create new directory (Terraform project).
2. Create S3 buckets in regions: `us-east-1`, `us-west-2`.
3. Use provider aliases to manage different AWS regions.
4. Implement variables for customizing bucket names, regions, and replication settings.
5. Ensure each bucket has a unique name across all AWS regions.
6. Enable versioning for each bucket to support data recovery.
7. Set up a lifecycle rule to transition infrequently accessed objects to S3 Glacier Instant Retrieval 
   storage class after 90 days for cost optimization.
8. Create outputs for each bucket's ARN, region, and replication status.

Use code templates provided below as a starting point.

```hcl
# variables.tf

variable "regions" {
  type    = list(string)
  default = # define your regions
}

variable "bucket_name_prefix" {
  type    = string
  default = # define your bucket prefixes
}
```

```hcl
# main.tf

terraform {
  required_version = "~> 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }

    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = var.regions[0]
}

# define rest of providers, each for the regions you specified, do not forget about the alias option that must be a string
#...
```

```hcl
# s3.tf

resource "random_id" "bucket_suffix" {
  count       = length(var.regions) # notice new option - it will create N resources that can be accesses by [index]
  byte_length = # specify how many bytes you want
}

resource "aws_s3_bucket" "s3_us_east_1" {
  # concatenating strings: "${variable}-${other_variable}rest_of_string"
  # accessing different random_id formats: random_id.bucket_suffix[i].hex / int / ...

  bucket = # create bucket name concatenating bucket_name_prefix, region, and suffix using random_id in hex format
}

resource "aws_s3_bucket_versioning" "s3_us_east_1" {
  bucket = aws_s3_bucket.s3_us_east_1.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "s3_us_east_1" {
  bucket = aws_s3_bucket.s3_us_east_1.id

  rule {
    id     = "transition-to-glacier"
    status = "Enabled"

    transition {
      days          = # specify correct n days here
      storage_class = # specify correct storage_class (https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/s3_bucket_lifecycle_configuration)
    }
  }
}


# define rest of buckets and its configurations, each for the regions you specified, do not forget to pass valid provider to resource options as in aliases example.
#...
```

```hcl
# outputs.tf

output "bucket_arns" {
   value = {
      "${var.regions[0]}" = aws_s3_bucket.s3_us_east_1.arn,
      # define rest keys in outputs, each for the regions you specified
   }
}

output "bucket_regions" {
   value = {
      "${aws_s3_bucket.s3_us_east_1.id}"     = var.regions[0],
      # define rest keys in outputs, each for the regions you specified
   }
}
```

After successfully applying your configuration:
- **Document your deployment**: save the `terraform apply` output and/or a screenshot of the created
  S3 buckets in the AWS Console as proof of successful deployment.
- Run `terraform destroy` to remove all created resources and avoid unnecessary AWS costs.

## Modules

In the previous exercise, we have quite a significant code repetition. When performing
deployments for different environments (e.g. staging, production) or regions, we often end up
with very similar resource blocks, with just minor differences. This violates the DRY principle
and creates maintenance challenges. 

Modules help solve this problem by encapsulating related resources into reusable components.
This reduces code duplication, improves maintainability, and enables consistent deployment
across different environments and regions.

### Exercise 3 - S3 refactoring

Terraform modules have specific desired structure that is worth to respect to keep it consistent,
well documented (via code), and easily to manage.

The structure is currently as follows:
```
. / your root project directory
├── main.tf
├── variables.tf
├── outputs.tf
├── modules/
│   └── s3_bucket/
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
```

So let's do some refactoring:
1. Create `modules` directory. Create `s3_bucket` directory inside `/modules`.
2. Create `variables.tf` in `modules/s3_bucket` with appropriate descriptions:
   - bucket_name_prefix, string
   - region, string
   - random_suffix, string
   - lifecycle_days, number, default = 90
   - lifecycle_storage_class, string, default = "GLACIER"
3. Create `main.tf`:
```hcl
# modules/s3_bucket/main.tf

resource "aws_s3_bucket" "this" {
   bucket = "${var.bucket_name_prefix}-${var.region}-${var.random_suffix}"
}

resource "aws_s3_bucket_versioning" "this" {
   bucket = aws_s3_bucket.this.id

   versioning_configuration {
      status = "Enabled"
   }
}

resource "aws_s3_bucket_lifecycle_configuration" "this" {
   bucket = aws_s3_bucket.this.id

   rule {
      id     = "transition-to-glacier"
      status = "Enabled"

      transition {
         days          = var.lifecycle_days
         storage_class = var.lifecycle_storage_class
      }
   }
}
```
4. Create `outputs.tf`:
```hcl
# modules/s3_bucket/outputs.tf

output "bucket_id" {
  value = aws_s3_bucket.this.id
}

output "bucket_arn" {
  value = aws_s3_bucket.this.arn
}

output "bucket_region" {
  value = var.region
}
```

5. In the project root, define variables:
   - regions, list(string), default = `# our previously defined regions`
   - bucket_name_prefix, string, default = `# "multi-region-bucket"  or any other prefix you wish`

6. In `main.tf`, define terraform config:
   - required_version
   - required_providers

```hcl
# main.tf

terraform {
  required_version = "~> 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}
```

7. In `main.tf`, define providers exactly as in previous exercise:
```hcl
# main.tf

terraform {
  required_version = ...
  required_providers ...
}

provider "aws" {
  region = var.regions[0]
}

provider "aws" {
  alias  = "us_west_2"
  region = var.regions[1]
}

```

8. In the project root, create file `s3.tf` and define S3 resources using the new module:

```hcl
# s3.tf
resource "random_id" "bucket_suffix" {
  count       = length(var.regions) # notice new option - it will create N resources that can be accesses by [index]
  keepers = {
    # suffix regenerates only when bucket_name_prefix changes, not on every apply
    bucket_name_prefix = var.bucket_name_prefix
  }
  byte_length = 6
}

# create us-east-1 bucket using the default provider
module "s3_us_east_1" {
  source            = "./modules/s3_bucket"
  bucket_name_prefix = var.bucket_name_prefix
  region            = var.regions[0]
  random_suffix     = random_id.bucket_suffix[0].hex
}
# create us-west-2 bucket using the eu_west_1 provider
module "s3_us_west_2" {
  source            = "./modules/s3_bucket"
  # notice how we pass provider alias to the module.
  providers = {
    aws = aws.us_west_2
  }
  bucket_name_prefix = var.bucket_name_prefix
  region            = var.regions[1]
  random_suffix     = random_id.bucket_suffix[1].hex
  lifecycle_days = 30 # we can modify the days value if we want
}

```

9. In the project root, define outputs:

```hcl
# outputs.tf

output "bucket_arns" {
  value = {
    "${var.regions[0]}" = module.s3_us_east_1.bucket_arn,
    "${var.regions[1]}" = module.s3_us_west_2.bucket_arn,
  }
}

output "bucket_regions" {
  value = {
    "${module.s3_us_east_1.bucket_id}"     = var.regions[0],
    "${module.s3_us_west_2.bucket_id}"     = var.regions[1],
  }
}
```

Now review the changes and determine for yourself whether
it is now clearer, shorter and easier to maintain the module.

After successfully applying your configuration:
- **Document your deployment**: save the `terraform apply` output and/or a screenshot of the created
  S3 buckets in the AWS Console as proof of successful deployment.
- Run `terraform destroy` to remove all created resources and avoid unnecessary AWS costs.

## Grading [15 points]

1. GitHub repository creation [2 points]
2. GitHub repository refactor [3 points]
3. Aliases [2 points]
4. S3 buckets across multiple regions [4 points]
5. Modules - S3 setup refactor [4 points]
