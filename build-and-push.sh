#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-ap-southeast-1}"
REPO_PREFIX="${REPO_PREFIX:-mlops-prod}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

declare -A TAGS
for arg in "$@"; do
  if [[ "$arg" == *=* ]]; then
    name="${arg%%=*}"
    val="${arg#*=}"
    TAGS["$name"]="$val"
  fi
done

IMAGES=("api" "producer" "consumer" "drift-monitor" "webhook")

echo "=== Logging in to ECR ==="
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "$REGISTRY"

for img in "${IMAGES[@]}"; do
  tag="${TAGS[$img]:-latest}"
  repo_name="${REPO_PREFIX}-${img}"
  local_tag="$repo_name:$tag"
  remote_uri="$REGISTRY/$local_tag"

  dockerfile_flag=""
  if [[ -f "Dockerfile.$img" ]]; then
    dockerfile_flag="-f Dockerfile.$img"
  fi

  echo "=== Building $local_tag ==="
  docker build $dockerfile_flag --target "$img" -t "$local_tag" .

  echo "=== Tagging & pushing $img ==="
  docker tag "$local_tag" "$remote_uri"
  docker push "$remote_uri"
done

echo "=== Done ==="
