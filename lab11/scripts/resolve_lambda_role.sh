#!/usr/bin/env bash
# Resolve the Lambda execution role ARN from a role name.
# Writes role_arn=<arn> to $GITHUB_OUTPUT for later workflow steps.
# Required env: ROLE_NAME (default: LabRole), GITHUB_OUTPUT (provided by GitHub Actions)
set -euo pipefail

ROLE_NAME="${ROLE_NAME:-LabRole}"
ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)

echo "Resolved role ARN: $ROLE_ARN"
echo "role_arn=$ROLE_ARN" >> "$GITHUB_OUTPUT"
