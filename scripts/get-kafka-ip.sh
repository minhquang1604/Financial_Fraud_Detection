#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-ap-southeast-1}"
ECS_CLUSTER="mlops-prod"

echo "=== Getting Kafka public IP ==="

TASK_ARN=$(aws ecs list-tasks --cluster "$ECS_CLUSTER" --service-name "${ECS_CLUSTER}-kafka-kafka" --region "$AWS_REGION" --query 'taskArns[0]' --output text)

if [ -z "$TASK_ARN" ] || [ "$TASK_ARN" = "None" ]; then
  echo "No running Kafka task found"
  exit 1
fi

TASK_ID=$(echo "$TASK_ARN" | awk -F/ '{print $NF}')
echo "Task: $TASK_ID"

ENI=$(aws ecs describe-tasks --cluster "$ECS_CLUSTER" --tasks "$TASK_ID" --region "$AWS_REGION" --query 'tasks[0].attachments[?type==`ElasticNetworkInterface`]|[0].details[?name==`networkInterfaceId`].value' --output text)

PUBLIC_IP=$(aws ec2 describe-network-interfaces --network-interface-ids "$ENI" --region "$AWS_REGION" --query 'NetworkInterfaces[0].Association.PublicIp' --output text)

echo ""
echo "Kafka public IP: $PUBLIC_IP"
echo ""
echo "To use Kafka locally:"
echo ""
echo "  echo \"$PUBLIC_IP mlops-prod-kafka\" | sudo tee -a /etc/hosts"
echo ""
echo "  KAFKA_BOOTSTRAP_SERVERS=mlops-prod-kafka:9092 ./scripts/run-producer.sh"
