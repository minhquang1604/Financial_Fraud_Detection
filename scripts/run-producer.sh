#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Load .env ──────────────────────────────────────────
if [ -f "$PROJECT_DIR/.env" ]; then
  set -a
  . "$PROJECT_DIR/.env"
  set +a
fi

# ── Config ────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS="${KAFKA_BOOTSTRAP_SERVERS:-mlops-prod-kafka:9092}"
TOPIC_NAME="${TOPIC_NAME:-transaction_events}"
DELAY_SECONDS="${DELAY_SECONDS:-0.1}"

# ── Ensure Kafka is reachable ──────────────────────────
echo "Kafka: $KAFKA_BOOTSTRAP_SERVERS"

# ── Activate venv ────────────────────────────────────────
if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
  source "$PROJECT_DIR/venv/bin/activate"
fi

# ── Create topic (if local Kafka) ─────────────────────────
if [[ "$KAFKA_BOOTSTRAP_SERVERS" == *"localhost"* ]]; then
  echo "Creating topic '$TOPIC_NAME' (if not exists)..."
  kafka-topics --create --if-not-exists \
    --bootstrap-server "$KAFKA_BOOTSTRAP_SERVERS" \
    --replication-factor 1 --partitions 1 \
    --topic "$TOPIC_NAME" 2>/dev/null || true
fi

# ── Run producer ─────────────────────────────────────────
echo "Starting producer (KAFKA=$KAFKA_BOOTSTRAP_SERVERS, TOPIC=$TOPIC_NAME)"
echo ""

exec python3 "$PROJECT_DIR/src/streaming/producer.py"
