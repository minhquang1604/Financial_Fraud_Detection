#!/usr/bin/env bash
set -euo pipefail

MLFLOW_URL="http://mlops-prod-1098509742.ap-southeast-1.elb.amazonaws.com:5000"
PGHOST="mlops-prod-mlflow.c9kiem2c2yw1.ap-southeast-1.rds.amazonaws.com"
PGPORT="5432"
PGUSER="mlflow_admin"
PGPASSWORD="SuperSecretPassword123!"
PGDATABASE="mlflow"

echo "=== Step 1: Soft-delete models via API ==="
MODELS=$(curl -s "${MLFLOW_URL}/api/2.0/mlflow/registered-models/search" \
  | python3 -c "import sys,json; print('\n'.join(m['name'] for m in json.load(sys.stdin).get('registered_models',[])))")

for name in $MODELS; do
  echo "  Model: $name"
  VERSIONS=$(curl -s "${MLFLOW_URL}/api/2.0/mlflow/model-versions/search?filter=name%3D%27${name}%27" \
    | python3 -c "import sys,json; print('\n'.join(v['version'] for v in json.load(sys.stdin).get('model_versions',[])))")
  for ver in $VERSIONS; do
    echo "    deleting v${ver}..."
    curl -s -X DELETE "${MLFLOW_URL}/api/2.0/mlflow/model-versions/delete" \
      -H "Content-Type: application/json" \
      -d "{\"name\":\"${name}\",\"version\":\"${ver}\"}" > /dev/null
  done
  curl -s -X DELETE "${MLFLOW_URL}/api/2.0/mlflow/registered-models/delete" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"${name}\"}" > /dev/null
done

echo ""
echo "=== Step 2: Soft-delete experiments via API ==="
EXPERIMENTS=$(curl -s "${MLFLOW_URL}/api/2.0/mlflow/experiments/search?max_results=1000" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
for e in data.get('experiments', []):
    if e['lifecycle_stage'] == 'active' and e['name'] != 'Default':
        print(e['experiment_id'])
")

for eid in $EXPERIMENTS; do
  echo "  Experiment $eid"
  curl -s -X POST "${MLFLOW_URL}/api/2.0/mlflow/experiments/delete" \
    -H "Content-Type: application/json" \
    -d "{\"experiment_id\":\"${eid}\"}" > /dev/null
done

echo ""
echo ""
echo "=== Done ==="
echo "Models and experiments are soft-deleted (hidden from UI)."
echo "To permanently purge, connect to RDS from inside the VPC and run:"
echo "  psql -h $PGHOST -U $PGUSER -d $PGDATABASE"
echo "  (tables: model_version_tags, model_versions, registered_model_tags, registered_models,"
echo "   tags, latest_metrics, metrics, params, runs, experiments)"
