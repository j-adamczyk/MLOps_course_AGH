#!/usr/bin/env bash
# Deploy the SAM stack with the freshly built image.
# Required env: STACK_NAME, AWS_REGION, IMAGE_REPOSITORY, IMAGE_URI, LAMBDA_ROLE_ARN
set -euo pipefail

sam deploy \
  --template-file sam-template.yaml \
  --stack-name "$STACK_NAME" \
  --image-repository "$IMAGE_REPOSITORY" \
  --parameter-overrides \
    "ImageUri=$IMAGE_URI" \
    "LambdaExecutionRoleArn=$LAMBDA_ROLE_ARN" \
  --capabilities CAPABILITY_IAM \
  --no-confirm-changeset \
  --no-fail-on-empty-changeset \
  --region "$AWS_REGION"
