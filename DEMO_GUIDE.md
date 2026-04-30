# Fraud Detection Demo - Auto Retrain Pipeline

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     AUTO RETRAIN PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │   STAGING    │───▶│  PREDICTED   │───▶│   LABELED    │        │
│  │ 20% Test Data│    │  (Live Data) │    │  (With Class) │        │
│  │  (no label)  │    │              │    │              │        │
│  └──────────────┘    └──────────────┘    └──────────────┘        │
│         │                   │                   │               │
│         │                   ▼                   ▼               │
│         │           ┌─────────────────────┐                   │
│         │           │     DRIFT MONITOR    │                   │
│         │           │   (PSI > 0.1)       │───▶ Webhook      │
│         │           └─────────────────────┘                   │
│         │                   │                                │
│         │                   ▼                                │
│         │           ┌─────────────────────┐                   │
│         │           │  GITHUB ACTIONS    │                   │
│         │           │  (retrain.yml)     │                   │
│         │           └─────────────────────┘                   │
│         │                   │                                │
│         ▼                   ▼                                │
│  ┌──────────────────────────────────────────┐               │
│  │         PREPARE DATA (Secret Sauce)       │               │
│  │      75% Ref + 25% Live (tránh forget)    │               │
│  └──────────────────────────────────────────┘               │
│         │                                                     │
│         ▼                                                     │
│  ┌──────────────────────────────────────────┐               │
│  │   RETRAIN PIPELINE (with SMOTE)            │               │
│  │        Model V2 → MLflow                   │               │
│  └──────────────────────────────────────────┘               │
│                                                                     │
└─────────────────────────────────────────────────────────────────���───┘
```

## Directory Structure

```
fraud-detection-project/
├── src/
│   ├── api/main.py              # FastAPI - predict + lưu live
│   ├── pipeline/
│   │   ├── prepare_data.py     # Mix 75% Ref + 25% Live
│   │   ├── retrain_pipeline.py # Train với SMOTE
│   │   ├── label_joiner.py     # Gán nhãn từ dataset gốc
│   │   └── auto_label_joiner.py # Auto watch staging
│   ├── monitoring/
│   │   ├── drift_detector.py  # PSI metric
│   │   └── auto_drift_monitor.py # Monitor + webhook
│   └── train/
├── data/
│   ├── raw/creditcard.csv     # Dataset gốc
│   ├── labeled/               # Reference data (80%)
│   ├── staging/               # Test data (20% - chưa có label)
│   ├── live/                  # Predictions (chưa có Class)
│   └── mixed/                 # Mixed data cho retrain
├── model/
│   └── fraud_model.pkl        # Model hiện tại
├── monitoring/
│   ├── prometheus.yml
│   └── grafana-dashboards/
├── .github/workflows/
│   └── retrain.yml           # GitHub Actions workflow
├── docker-compose.mlops.yml  # Full stack (Kafka, API, MLflow...)
└── requirements.txt
```

## Demo Commands

### Bước 1: Tạo 20% test data → staging (không có Class)

```bash
python -c "
import pandas as pd
import numpy as np

# Load dataset gốc
df = pd.read_csv('data/raw/creditcard.csv')
df = df.sample(frac=0.2, random_state=42)

# Xóa cột Class (giả lập chưa có label)
staging_df = df.drop('Class', axis=1)

# Lưu vào staging
staging_df.to_parquet('data/staging/staging_batch_v1.parquet', index=False)
print(f'Tạo staging: {len(staging_df)} records')
"
```

### Bước 2: Start Services

```bash
# Terminal 1: Start Zookeeper + Kafka
docker-compose -f docker-compose.mlops.yml up -d zookeeper kafka

# Terminal 2: Start FastAPI
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
```

### Bước 3: Start Producer + Consumer

```bash
# Terminal 3: Start Consumer (đọc Kafka → gọi API → lưu live)
python -m src.streaming.consumer
```

### Bước 4: Send Data to Kafka

```bash
# Terminal 4: Start Producer (gửi từ staging data)
python -m src.streaming.producer
```

**Output:**
```
Loaded 56962 records from data/staging/staging_batch_v1.parquet
Producer connected to localhost:9092
Publishing messages to topic 'transaction_events'...
Sent: txn_0 -> Partition: 0, Offset: 0
Sent: txn_1 -> Partition: 0, Offset: 1
...
```

**Consumer output:**
```
Consumer connected to localhost:9092
Listening to topic: transaction_events
------------------------------------------------------------
[txn_0] Time: 1234 | Prediction: 0 | Probability: 0.1234 | Normal transaction
[txn_1] Time: 1235 | Prediction: 0 | Probability: 0.0876 | Normal transaction
...
```

**Kết quả:** Tất cả predictions được lưu vào `data/live/live_predictions.parquet`

### Bước 5: Label Joiner (gán Class từ dataset gốc)

```bash
python -m src.pipeline.label_joiner
```

**Output:**
```
============================================================
LABEL JOINER RESULT
============================================================
Staging: data/staging/staging_batch_v1.parquet
Output: data/labeled/labeled_batch_20240430_123456.parquet
Records: 56962
Class 0: 56924
Class 1: 38
============================================================
```

### Bước 6: Drift Monitor (phát hiện drift)

```bash
# Với interval ngắn để test nhanh
python -m src.monitoring.auto_drift_monitor \
  --reference data/labeled/labeled_batch_v0001.parquet \
  --model model/fraud_model.pkl \
  --interval 30
```

**Output:**
```
============================================================
AUTO DRIFT MONITOR + RETRAIN
Interval: 30s
Auto retrain: True
============================================================
Running drift check...
PSI drift detected: 0.15 > 0.1 threshold!
Webhook sent successfully: 200
Drift alert: alert_triggered=True
```

### Bước 7: GitHub Actions (tự động retrain)

**Tự động trigger** hoặc **manual**:

```bash
# Manual trigger
gh workflow run retrain.yml -f trigger_type=drift_alert
```

### Bước 8: Kiểm tra kết quả

```bash
# Xem mixed data
ls -la data/mixed/

# Xem model mới
ls -la model/

# Xem MLflow
# http://localhost:5000
```

## Auto Mode (chạy liên tục)

```bash
# Terminal 1: Drift Monitor
python -m src.monitoring.auto_drift_monitor \
  --reference data/labeled/labeled_batch_v0001.parquet \
  --model model/fraud_model.pkl \
  --interval 60

# Terminal 2: Auto Label Joiner (watch staging folder)
python -m src.pipeline.auto_label_joiner --mode watch
```

Khi drift phát hiện → webhook → GitHub Actions → retrain tự động.

## Environment Variables

```bash
# Webhook URL cho GitHub Actions (tùy chọn)
# Đặt trong GitHub Settings → Secrets → Actions
# - DRIFT_WEBHOOK_URL: https://api.github.com/repos/{owner}/{repo}/actions/workflows/retrain.yml/dispatch
# - PAT_TOKEN: (Personal Access Token có quyền workflow)

# Local test:
export DRIFT_WEBHOOK_URL="https://api.github.com/repos/{owner}/{repo}/actions/workflows/retrain.yml/dispatch"
export PAT_TOKEN="ghp_xxxx"  # Thay bằng PAT của bạn

# MLflow Tracking URI
export MLFLOW_TRACKING_URI="http://localhost:5000"
```
```