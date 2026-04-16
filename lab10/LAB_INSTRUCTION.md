# Laboratory 10 - Infrastructure as Code (IaC)

## Introduction

Infrastructure as Code (IaC) means managing and provisioning cloud infrastructure through code
rather than manual processes. This gives you automation, consistency across environments,
version control over your infrastructure, and the ability to review and roll back changes just
like application code.

To pass, submit:
- a link to a public GitHub repository with your IaC code
- a report (PDF or DOCX) with screenshots showing the created resources in AWS/GitHub UI and
  tool logs confirming the resources were provisioned by the tool, not created manually

## IaC tools

**[Terraform](https://www.terraform.io)** is the most popular IaC tool, using a declarative approach with its HCL language.
It is cloud-agnostic and has a large provider ecosystem covering AWS, GCP, Azure, GitHub, and more.

**[Pulumi](https://www.pulumi.com)** uses general-purpose programming languages (Python, TypeScript, Go) instead of a
domain-specific language. A good choice if you prefer writing code over configuration files.

AWS vendor-specific solutions: **[CloudFormation](https://aws.amazon.com/cloudformation/)** (JSON/YAML templates) and
**[AWS CDK](https://aws.amazon.com/cdk/)** (Python/TypeScript, compiles down to CloudFormation). Both risk vendor
lock-in; equivalent tools exist for GCP and Azure.

**[OpenTofu](https://opentofu.org)** is an open-source fork of Terraform, created in 2023 after HashiCorp changed
Terraform's license from MPL to BSL (Business Source License). Drop-in replacement with identical
syntax, maintained by the Linux Foundation. Worth knowing as many companies are evaluating or
already migrating to it.

**[Terragrunt](https://terragrunt.gruntwork.io)** is a thin wrapper around Terraform that reduces configuration
duplication across environments (e.g. staging, production). Not a replacement - used alongside
Terraform in larger projects.

**[SkyPilot](https://skypilot.readthedocs.io)** is an ML-focused tool for running training jobs and inference on any
cloud. You define compute requirements (GPU type, spot vs on-demand) and SkyPilot picks the
cheapest available option across AWS, GCP, Azure, and others, provisions the resources, runs the
job, and tears everything down. Increasingly common in ML infrastructure.

## Choose your tool

Both tracks cover the same infrastructure scenarios (GitHub repository management, S3 multi-region
buckets, code reuse patterns) so the learning outcomes are equivalent.

**Terraform** - if you prefer a declarative, configuration-based approach with a mature ecosystem
and wide industry adoption. Uses HCL, a straightforward domain-specific language.
[![GitHub stars](https://img.shields.io/github/stars/hashicorp/terraform?style=social)](https://github.com/hashicorp/terraform)
See [LAB_INSTRUCTION_TERRAFORM.md](LAB_INSTRUCTION_TERRAFORM.md).

**Pulumi** - if you prefer writing Python over learning a new configuration language, or if you
want to use familiar programming constructs like loops, functions, and classes to define infrastructure.
[![GitHub stars](https://img.shields.io/github/stars/pulumi/pulumi?style=social)](https://github.com/pulumi/pulumi)
See [LAB_INSTRUCTION_PULUMI.md](LAB_INSTRUCTION_PULUMI.md).

Not sure? Read how others decided:
- [Pulumi vs Terraform - official comparison](https://www.pulumi.com/docs/concepts/vs/terraform/)
- [r/devops - Pulumi vs Terraform discussion](https://www.reddit.com/r/devops/search/?q=pulumi+vs+terraform&sort=top)
- [r/Terraform - community perspective](https://www.reddit.com/r/Terraform/search/?q=pulumi&sort=top)
