# ML Fraud Detection — Data Flow

## Architecture Overview

```
┌──────────┐    Kafka     ┌──────────┐   HTTP     ┌──────────┐   MLflow   ┌──────────┐
│ Producer │─────────────▶│ Consumer │───────────▶│   API    │──────────▶│  MLflow  │
│ (RunTask)│  txn_events  │ (service)│ POST:8000  │ (service)│   :5000   │ (service)│
└──────────┘              └────┬─────┘            └────┬─────┘           └────┬─────┘
                               │                        │                     │
                         saves to                 Prometheus              ┌──┴──┐
                            S3                     metrics                │ RDS │
                                                                         │ S3  │
                                                                         └─────┘

┌──────────────┐   webhook   ┌──────────┐   poll:300s   ┌────────────────┐
│ GitHub │◀───────┤ Drift   │◀─────────────│   Prometheus  │
│ Actions       │  (Alert)    │ Monitor  │               │    Grafana    │
└──────────────┘             │ (RunTask)│               └────────────────┘
       │                      └──────────┘
       │ retrain.yml
       ▼
[RetrainPipeline] ──log──▶ MLflow ──register──▶ Model Version

┌──────────┐
│ Webhook  │  (receives external alerts, dispatches GitHub Actions)
│ (service)│
└──────────┘
```

## Service DNS Names (ECS Service Connect)

| Service    | DNS Name                  | Port |
|------------|---------------------------|------|
| API        | `mlops-prod-api:8000`     | 8000 |
| MLflow     | `mlops-prod-mlflow:5000`  | 5000 |
| Kafka      | `mlops-prod-kafka:9092`   | 9092 |
| Zookeeper  | `mlops-prod-zookeeper:2181` | 2181 |
| Prometheus | `mlops-prod-prometheus:9090` | 9090 |
| Grafana    | `mlops-prod-grafana:3000` | 3000 |
| Webhook    | `mlops-prod-webhook:5001` | 5001 |

## Component Details

### 1. Producer (RunTask, not long-running)

- Reads historical transactions from `data/staging/staging_batch_v1.parquet`
- Publishes each row as JSON to Kafka topic `transaction_events`
- Features: 28 PCA components (V1–V28) + Amount + Time

### 2. Consumer (service)

- Subscribes to Kafka topic `transaction_events`
- For each message: calls `POST http://mlops-prod-api:8000/predict`
- Buffers results and saves as Parquet to S3 (`data/realtime/`)
- Configurable via env vars: `KAFKA_BOOTSTRAP_SERVERS`, `API_URL`, `USE_S3`, `S3_BUCKET_NAME`

### 3. API (service)

FastAPI application with blue-green model deployment.

**Endpoints:**

| Method | Path               | Description                              |
|--------|--------------------|------------------------------------------|
| GET    | `/health`          | Health check with model status           |
| GET    | `/metrics`         | Prometheus metrics                       |
| POST   | `/predict`         | Submit features, get fraud prediction    |
| GET    | `/model/info`      | Active/standby model details             |
| GET    | `/model/history`   | Last N model swaps                       |
| POST   | `/model/rollback`  | Rollback to previous model version       |
| POST   | `/model/update`    | Manually trigger model update            |

**Prediction flow:**
1. Validate input (V1–V28, Amount, Time)
2. Compute stats: `mean_amt`, `median_amt`, `threshold_95`
3. Feature engineering → 6 derived features:
   - `time_diff`, `time_diff_log`, `log_amount`
   - `amt_to_mean_ratio`, `amt_to_median_ratio`, `is_high_amount`
4. Select 34 features (28 PCA + 6 engineered)
5. Run XGBoost inference via `ModelRouter`
6. Apply decision threshold → fraud / normal
7. Record latency, predictions to Prometheus
8. Track success/failure in `RollbackManager`

**Model lifecycle (auto-update watcher):**
- Background thread polls MLflow every 60s for new model versions
- New version detected → loads into standby slot
- Validates: warmup, latency check, F1 comparison
- If pass → swap standby ↔ active (zero-downtime)

### 4. MLflow (service)

- Model registry + experiment tracking
- Backend store: PostgreSQL on RDS (runs, params, metrics)
- Artifact store: S3 bucket (model binaries)
- Queried by API for `FraudDetectionModel` in Production/Staging stage

### 5. Drift Monitor (RunTask, not long-running)

Runs every 300s:

1. Load reference data from `data/train/train_full.parquet`
2. Load current staging data from S3 (`data/realtime/`)
3. Feature engineering on both sets
4. Drift detection via Evidently AI / KS tests:
   - Data drift (feature distribution shifts)
   - Concept drift (performance degradation)
   - Prediction drift (output distribution)
5. Update Prometheus gauges on port 8001
6. If drift detected:
   - Send webhook → GitHub Actions (`retrain.yml`)
   - Commit/push new data files to Git
   - If auto-retrain enabled + cooldown passed + enough data:
     - Train new XGBoost model
     - Log to MLflow (params, metrics, model)
     - Register new version, transition to Production

### 6. Webhook (service)

- Flask server on port 5001
- Receives Alertmanager/drift webhooks
- Dispatches GitHub Actions workflow `retrain.yml` via PAT

### 7. Prometheus + Grafana

- Prometheus scrapes API `/metrics` (port 8000) and drift monitor metrics (port 8001)
- Grafana dashboards for visualization
- Both on Fargate Spot (cost-optimized)

## Externally Routed Paths (ALB)

| Path Pattern     | Target Service |
|------------------|----------------|
| `/api/*`         | API            |
| `/predict`       | API            |
| `/health`        | API            |
| `/metrics`       | API            |
| `/model/*`       | API            |
| `/mlflow/*`      | MLflow         |
| `/grafana/*`     | Grafana        |
| `/prometheus/*`  | Prometheus     |

## Infrastructure Layers

```
prod-infra (terraform/environments/prod-infra/)
├── VPC (public + private subnets, NAT gateway)
├── ECR repositories (api, consumer, producer, drift-monitor, webhook)
├── RDS PostgreSQL (MLflow backend)
├── ECS cluster (Fargate, Service Connect namespace)
├── IAM roles (execution + task)
├── ALB (public-facing, listener on port 80)
└── Security groups

prod-app (terraform/environments/prod-app/)
├── ECS services (8 services: mlflow, api, kafka, zookeeper,
│                  consumer, prometheus, grafana, webhook)
├── ECS task definitions
├── ALB listener rules
└── CloudWatch log groups
```

## S3 Bucket Layout

```
aws-terraform-remotebackend/
└── tfstate/
    └── mlops-project/
        ├── infra.tfstate
        └── app.tfstate

[mlops-prod S3 bucket]
├── mlflow/           (MLflow artifact store)
├── data/
│   ├── realtime/     (consumer batch output)
│   ├── labeled/      (LabelJoiner output)
│   ├── processed/    (archived data)
│   └── train/        (training datasets)
```
