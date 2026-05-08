#!/usr/bin/env bash
# Delete CloudFormation stack if it is in any *ROLLBACK* state.
# Required env: STACK_NAME, AWS_REGION
set -euo pipefail

STATUS=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --region "$AWS_REGION" \
  --query 'Stacks[0].StackStatus' \
  --output text 2>/dev/null || echo "NOT_FOUND")

echo "Current stack status: $STATUS"

case "$STATUS" in
  *ROLLBACK*)
    aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$AWS_REGION"
    aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME" --region "$AWS_REGION"
    ;;
esac
