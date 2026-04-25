# Real-time Credit Card Fraud Detection

A complete MLOps pipeline for credit card fraud detection using XGBoost, MLflow, FastAPI, and Kafka.

## Project Structure

```
fraud-detection-project/
├── data/
│   ├── raw/                  # Raw creditcard.csv
│   ├── processed/           # Cleaned parquet files
│   └── test/                 # Test dataset for Kafka
├── src/
│   ├── train/                # Model training
│   │   ├── train.py
│   │   └── utils.py
│   ├── api/                  # FastAPI service
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── model_loader.py
│   └── streaming/            # Kafka pipeline
│       ├── producer.py
│       └── consumer.py
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Start Infrastructure

```bash
docker-compose up -d
```

This starts:
- Kafka (port 9092)
- Zookeeper (port 2181)
- MLflow UI (port 5000)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model

```bash
cd src/train
python train.py
```

This will:
- Load and clean data from `data/raw/creditcard.csv`
- Engineer features
- Train XGBoost model
- Log metrics to MLflow
- Save test data to `data/test/test_data.parquet`

### 4. Start FastAPI Service

```bash
cd src/api
python main.py
```

API runs at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

### 5. Run Kafka Producer

```bash
cd src/streaming
python producer.py
```

Reads test data and publishes to Kafka topic `transaction_events`.

### 6. Run Kafka Consumer

```bash
cd src/streaming
python consumer.py
```

Consumes messages, calls FastAPI `/predict`, and prints results.

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Fraud prediction

## Environment Variables

- `KAFKA_BOOTSTRAP_SERVERS` - Kafka broker address (default: `localhost:9092`)
- `MLFLOW_TRACKING_URI` - MLflow server URI (default: `http://localhost:5000`)
- `API_URL` - FastAPI URL (default: `http://localhost:8000`)