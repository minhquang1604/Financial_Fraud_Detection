# Fraud Detection System

MLOps pipeline cho fraud detection sử dụng XGBoost, Kafka, MLflow, Prometheus & Grafana.

```
Producer → Kafka → Consumer → S3 → Label → Train → S3 → API
                              ↓
                        Monitoring (Prometheus/Grafana)
                              ↓
                        GitHub Actions (Auto Retrain)
```

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Producer    │────▶│    Kafka     │────▶│  Consumer    │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                    ┌───────────────────────────┴───────────────────────────┐
                    ▼                           ▼                           ▼
             ┌──────────────┐          ┌──────────────┐           ┌──────────────┐
             │     S3        │          │  API (FastAPI)│           │    S3        │
             │ (data/realtime)│          │  localhost:8000│          │ (live predictions)
             └──────────────┘          └──────────────┘           └──────────────┘
                    │                                                    
                    ▼                                                    
             ┌──────────────┐                                         
             │ Label Joiner │                                         
             │ (gán nhãn)   │                                         
             └──────┬───────┘                                         
                    ▼                                                    
             ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
             │     S3        │────▶│ GitHub Actions│────▶│   MLflow     │
             │ (data/labeled)│     │  (retrain)   │     │  (metrics)   │
             └──────────────┘     └──────────────┘     └──────────────┘
```

## Quick Start

### 1. Start Infrastructure

```bash
docker-compose up -d
```

Services:
| Service | Port | URL |
|---------|------|-----|
| Kafka | 9092 | localhost:9092 |
| Zookeeper | 2181 | localhost:2181 |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 (admin/admin) |
| AlertManager | 9093 | http://localhost:9093 |
| Webhook Receiver | 5001 | http://localhost:5001 |

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model

```bash
python -m src.train.train
```

### 4. Start API

```bash
python -m src.api.main
# API docs: http://localhost:8000/docs
```

### 5. Run Streaming Pipeline

```bash
# Terminal 1: Producer (gửi data vào Kafka)
python -m src.streaming.producer

# Terminal 2: Consumer (đọc từ Kafka → predict → upload S3)
python -m src.streaming.consumer
```

### 6. Label & Retrain

```bash
# Gán nhãn cho realtime data
python -m src.pipeline.label_joiner

# Mix data + Retrain
python -m src.pipeline.prepare_data
python -m src.pipeline.retrain_pipeline
```

## Auto Retrain (100% Webhook)

Drift detection → Webhook → GitHub Actions → Auto retrain

```bash
# Trigger manual retrain
gh workflow run retrain.yml -f trigger_type=manual

# Hoặc trigger từ GitHub UI
# https://github.com/minhquang1604/Financial_Fraud_Detection/actions
```

**GitHub Secrets cần thiết:**
- `MLFLOW_TRACKING_URI` - MLflow server URL
- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `AWS_DEFAULT_REGION` - AWS region (vd: us-east-1)
- `S3_BUCKET_NAME` - S3 bucket name

## API Usage

```bash
# Predict
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
      "Amount": 150.0, "Time": 50000
    }
  }'
```

## Project Structure

```
fraud-detection-project/
├── src/
│   ├── api/                    # FastAPI service
│   │   └── main.py            # /predict endpoint
│   ├── streaming/             # Kafka pipeline
│   │   ├── producer.py        # Gửi data vào Kafka
│   │   └── consumer.py         # Đọc từ Kafka → predict → S3
│   ├── pipeline/              # Data pipeline
│   │   ├── label_joiner.py    # Gán nhãn từ dataset gốc
│   │   ├── prepare_data.py     # Mix data (75% ref + 25% new)
│   │   └── retrain_pipeline.py # Retrain với SMOTE
│   ├── monitoring/            # Drift detection
│   │   └── auto_drift_monitor.py # PSI monitoring + webhook
│   ├── train/                # Training utilities
│   │   └── utils.py           # Feature engineering
│   └── mlops/                # MLOps utilities
│       ├── s3_manager.py     # S3 operations
│       └── data_version.py   # Data versioning
├── monitoring/               # Monitoring config
│   ├── prometheus.yml
│   └── grafana-dashboards/
├── model/                    # Trained models (.pkl)
├── data/                     # Data files (gitignored)
│   ├── raw/                  # creditcard.csv (reference)
│   ├── train/                # Training data
│   ├── realtime/             # Live data from Kafka
│   └── labeled/              # Labeled data
├── docker-compose.yml        # All services
└── requirements.txt
```

## Environment Variables

```bash
# S3
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=your-bucket-name
USE_S3=true

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# MLflow
MLFLOW_TRACKING_URI=http://13.229.113.113:5000

# Webhook (for auto retrain)
DRIFT_WEBHOOK_URL=https://api.github.com/repos/owner/repo/actions/workflows/retrain.yml/dispatch
PAT_TOKEN=ghp_your_token
```

## Data Flow

1. **Streaming**: `creditcard.csv (staging)` → Kafka → Consumer → **S3 (realtime/)**
2. **Labeling**: **S3 (realtime/)** → Label Joiner → **S3 (labeled/)**
3. **Training**: **S3 (labeled/)** + train data → Prepare → Retrain → **S3 (model/)**
4. **Monitoring**: Prometheus metrics → Grafana dashboard

## Monitoring

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **AlertManager**: http://localhost:9093

Alerts configured:
- High fraud prediction ratio
- Model drift detected (PSI > 0.1)
- Data drift detected

## License

MIT