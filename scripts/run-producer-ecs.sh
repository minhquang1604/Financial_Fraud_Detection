#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-ap-southeast-1}"
ECS_CLUSTER="mlops-prod"
TASK_FAMILY="${ECS_CLUSTER}-producer"
TASK_DEF=$(aws ecs describe-task-definition \
  --task-definition "$TASK_FAMILY" \
  --region "$AWS_REGION" \
  --query "taskDefinition.taskDefinitionArn" \
  --output text)

echo "Task definition: $TASK_DEF"

# Grab network config from a running service (consumer) to use same subnets + SG
SUBNETS=$(aws ecs describe-services \
  --cluster "$ECS_CLUSTER" \
  --services "${ECS_CLUSTER}-consumer" \
  --region "$AWS_REGION" \
  --query 'services[0].networkConfiguration.awsvpcConfiguration.subnets' \
  --output text | tr '\t' ',')

SG=$(aws ecs describe-services \
  --cluster "$ECS_CLUSTER" \
  --services "${ECS_CLUSTER}-consumer" \
  --region "$AWS_REGION" \
  --query 'services[0].networkConfiguration.awsvpcConfiguration.securityGroups[0]' \
  --output text)

echo "Subnets: $SUBNETS"
echo "Security group: $SG"

# Run the task
TASK_ARN=$(aws ecs run-task \
  --cluster "$ECS_CLUSTER" \
  --task-definition "$TASK_DEF" \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG]}" \
  --region "$AWS_REGION" \
  --query 'tasks[0].taskArn' \
  --output text)

TASK_ID=$(echo "$TASK_ARN" | awk -F/ '{print $NF}')
echo ""
echo "Started task: $TASK_ID"
echo "Streaming logs (Ctrl+C to stop)..."
echo ""

# Tail the logs until the task stops
LOG_GROUP="/ecs/${TASK_FAMILY}"
aws logs tail "$LOG_GROUP" --region "$AWS_REGION" --follow --since 0s &
TAIL_PID=$!

# Wait for the task to stop
aws ecs wait tasks-stopped \
  --cluster "$ECS_CLUSTER" \
  --tasks "$TASK_ID" \
  --region "$AWS_REGION" > /dev/null 2>&1 || true

kill "$TAIL_PID" 2>/dev/null || true
wait "$TAIL_PID" 2>/dev/null || true

echo ""
EXIT_CODE=$(aws ecs describe-tasks \
  --cluster "$ECS_CLUSTER" \
  --tasks "$TASK_ID" \
  --region "$AWS_REGION" \
  --query 'tasks[0].containers[0].exitCode' \
  --output text)

echo "Task $TASK_ID finished with exit code $EXIT_CODE"
