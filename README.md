# Fraud Detection System

A complete end-to-end MLOps pipeline for real-time credit card fraud detection using XGBoost, MLflow, FastAPI, Kafka, Prometheus & Grafana.

## Project Overview

This system provides:
- Real-time fraud prediction via REST API
- Kafka-based streaming pipeline
- Automated model retraining with drift monitoring
- Data versioning and label management
- Prometheus metrics & Grafana dashboards

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Producer  │───▶│   Kafka    │───▶│ Staging    │
└─────────────┘    └─────────────┘    └───────────┘
                      │                   │
                      ▼                   ▼
                 ┌─────────────┐    ┌─────────────┐
                 │  Consumer  │───▶│  API       │
                 └─────────────┘    └───────────┘
                                       │
                                       ▼
                                  ┌───────────┐
                                  │ Monitoring│
                                  │ + Drift  │
                                  └─────────┘
```

## Project Structure

```
fraud-detection-project/
├── data/
│   ├── raw/                  # Raw creditcard.csv
│   ├── processed/           # Cleaned training data
│   ├── labeled/           # Labeled data (reference + live-joined)
│   ├── staging/           # Raw Kafka messages
│   ├── live/             # Live predictions
│   ├── mixed/            # Mixed training data
│   └── dvc/             # Data version metadata
├── model/                 # Trained models (.pkl)
├── src/
│   ├── train/             # Model training
│   │   ├── train.py      # XGBoost training with MLflow
│   │   └── utils.py     # Feature engineering
│   ├── api/              # FastAPI service
│   │   ├── main.py      # Prediction API
│   │   ├── schemas.py   # Pydantic schemas
│   │   └── model_loader.py
│   ├── streaming/        # Kafka pipeline
│   │   ├── producer.py  # Publish to Kafka
│   │   └── consumer.py # Consume + predict
│   ├── staging/          # Staging consumer
│   │   └── staging_consumer.py
│   ├── pipeline/         # Data pipelines
│   │   ├── label_joiner.py       # Join labels from reference
│   │   ├── prepare_data.py       # Mix labeled + live data
│   │   ├── retrain_pipeline.py   # Automated retraining
│   │   └── auto_label_joiner.py  # Auto label joining
│   ├── monitoring/        # Monitoring & drift
│   │   ├── drift_detector.py      # PSI + KS drift detection
│   │   ├── auto_drift_monitor.py # Auto retrain trigger
│   │   ├── monitoring_server.py   # Prometheus exporter
│   │   └── metrics_exporter.py
│   └── mlops/            # MLOps utilities
│       ├── data_version.py      # Data versioning
│       └── data_split.py       # Train/test split
├── monitoring/            # Monitoring config
│   ├── prometheus.yml
│   ├── grafana-dashboards/
│   └── alerts.yml
├── docker-compose.yml       # Kafka + Zookeeper
├── docker-compose.mlops.yml  # MLflow
└── requirements.txt
```

## Quick Start

### 1. Start Infrastructure

```bash
# Kafka + Zookeeper
docker-compose up -d

# Or with MLflow
docker-compose -f docker-compose.mlops.yml up -d
```

Services:
| Service | Port |
|---------|------|
| Kafka | 9092 |
| Zookeeper | 2181 |
| MLflow UI | 5000 |
| Prometheus | 9090 |
| Grafana | 3000 |
| Alertmanager | 9093 |

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Data Split (Optional)

```bash
python -m src.mlops.data_split --train-ratio 0.8
```

Splits raw data into train (80%) and Kafka stream (20%).

### 4. Train Model

```bash
python -m src.train.train
```

This will:
- Load & clean data using DuckDB
- Engineer features
- Train XGBoost with SMOTE
- Log to MLflow
- Save model to `model/fraud_model.pkl`

### 5. Start API

```bash
python -m src.api.main
```

API at `http://localhost:8000`
- `GET /health` - Health check
- `POST /predict` - Fraud prediction
- Docs: `http://localhost:8000/docs`

### 6. Run Streaming Pipeline

```bash
# Terminal 1: Producer
python -m src.streaming.producer

# Terminal 2: Consumer (with API)
python -m src.streaming.consumer
```

### 7. Monitoring

```bash
# Start drift monitor
python -m src.monitoring.auto_drift_monitor \
    --reference data/processed/cleaned_fraud_data.parquet \
    --model model/fraud_model.pkl
```

## API Usage

### Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "V1": -1.34, "V2": 0.45, "V3": 1.23, "V4": -0.89,
      "V5": -0.34, "V6": 0.12, "V7": 0.45, "V8": -0.78,
      "V9": 0.23, "V10": -0.45, "V11": 1.23, "V12": -0.89,
      "V13": -0.34, "V14": 0.12, "V15": 0.45, "V16": -0.78,
      "V17": 0.23, "V18": -0.45, "V19": 1.23, "V20": -0.89,
      "V21": -0.34, "V22": 0.12, "V23": 0.45, "V24": -0.78,
      "V25": 0.23, "V26": -0.45, "V27": 1.23, "V28": -0.89,
      "Amount": 150.0,
      "Time": 50000
    }
  }'
```

Response:
```json
{
  "transaction_time": 50000,
  "fraud_probability": 0.0234,
  "prediction": 0,
  "message": "Normal transaction"
}
```

## Pipeline Commands

### Label Joiner

Join staging data with reference labels:

```bash
python -m src.pipeline.label_joiner --staging-file data/staging/staging_batch_v1.parquet
```

### Auto Label Joiner

Watch staging folder and auto-join:

```bash
python -m src.pipeline.auto_label_joiner --mode watch
```

### Prepare Mixed Data

Mix labeled reference + live data for retraining:

```bash
python -m src.pipeline.prepare_data --ref-ratio 0.75
```

### Retrain

```bash
python -m src.pipeline.retrain_pipeline --version v0001
```

## Monitoring

### Metrics Endpoint

Prometheus metrics at `http://localhost:8001`

### Grafana Dashboard

Access at `http://localhost:3000` (admin/admin)

### Alerts

Configured in `monitoring/alerts.yml`:
- High fraud prediction ratio
- Model drift detected
- Data drift (PSI > 0.1)

## Environment Variables

| Variable | Default | Description |
|----------|---------|------------|
| `KAFKA_BOOTSTRAP_SERVERS` | localhost:9092 | Kafka broker |
| `MLFLOW_TRACKING_URI` | http://localhost:5000 | MLflow server |
| `API_URL` | http://localhost:8000 | API URL |
| `PROMETHEUS_PORT` | 8001 | Prometheus port |

## Data Flow

1. **Training Flow**: Raw CSV → Cleaned → Labeled → Mixed → Train → Model
2. **Prediction Flow**: Live → API → Predictions saved to `data/live/`
3. **Label Flow**: Staging → Label Joiner → `data/labeled/`
4. **Retrain Flow**: Labeled + Live → Prepare → Retrain → New Model

## Features

- PCA-transformed features (V1-V28)
- Time-based features (time_diff, time_diff_log)
- Amount features (log_amount, amt ratios, is_high_amount)

## Model Metrics

Logged to MLflow:
- AUPRC
- F1 Score
- Precision / Recall
- Best threshold (tuned on validation set)