#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-ap-southeast-1}"
REPO_PREFIX="${REPO_PREFIX:-mlops-prod}"
TAG="${1:-latest}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

echo "=== Logging in to ECR ==="
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "$REGISTRY"

echo "=== Building api image ==="
docker build --target api -t "$REPO_PREFIX-api:$TAG" .

echo "=== Pushing to ECR ==="
docker tag "$REPO_PREFIX-api:$TAG" "$REGISTRY/$REPO_PREFIX-api:$TAG"
docker push "$REGISTRY/$REPO_PREFIX-api:$TAG"

echo "=== Done ==="
