#!/usr/bin/env bash
set -euo pipefail

ALB_DNS="mlops-prod-1098509742.ap-southeast-1.elb.amazonaws.com"
AWS_REGION="ap-southeast-1"
ECS_CLUSTER="mlops-prod"
API_SERVICE="mlops-prod-api"
MODEL_NAME="FraudDetectionModel"

VERSION="${1:-latest}"

if [ "$VERSION" = "latest" ]; then
  VERSION=$(curl -s "http://${ALB_DNS}:5000/api/2.0/mlflow/model-versions/search?filter=name%3D%27${MODEL_NAME}%27" \
    | python3 -c "import sys,json; vs=json.load(sys.stdin).get('model_versions',[]); print(max(int(v['version']) for v in vs))")
fi

echo "=== Promoting ${MODEL_NAME} v${VERSION} to Production ==="

curl -s -X POST "http://${ALB_DNS}:5000/api/2.0/mlflow/model-versions/transition-stage" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"${MODEL_NAME}\",\"version\":\"${VERSION}\",\"stage\":\"Production\"}" \
  | python3 -m json.tool

echo ""
echo "=== Redeploying API ==="
aws ecs update-service --cluster "$ECS_CLUSTER" --service "$API_SERVICE" \
  --force-new-deployment --region "$AWS_REGION" --query 'service.serviceName' --output text

echo "=== Waiting for healthy API ==="
for i in $(seq 1 12); do
  sleep 5
  STATUS=$(curl -s "http://${ALB_DNS}/health" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))")
  MODEL=$(curl -s "http://${ALB_DNS}/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('active_model',{}).get('version',''))")
  if [ "$STATUS" = "healthy" ]; then
    echo "API healthy, model v${MODEL} loaded"
    exit 0
  fi
  echo "  waiting... ($i)"
done

echo "Timed out waiting for healthy API"
exit 1
