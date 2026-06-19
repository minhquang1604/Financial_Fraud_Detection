FROM python:3.10-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONPATH=/app


FROM base AS api

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM base AS producer

CMD ["python", "-m", "src.streaming.producer"]

FROM base AS consumer

CMD ["python", "-m", "src.streaming.consumer"]


FROM base AS drift-monitor

CMD ["python", "-m", "src.monitoring.auto_drift_monitor", "--model", "mlflow:Production", "--interval", "300"]


FROM base AS webhook

RUN pip install --no-cache-dir flask

CMD ["python", "-m", "src.api.webhook_receiver"]
