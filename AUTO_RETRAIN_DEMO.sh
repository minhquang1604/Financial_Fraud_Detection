#!/bin/bash
# AUTO RETRAIN DEMO - 100% Webhook Triggered
# ==========================================

set -e

source venv/bin/activate

echo "=============================================="
echo "  AUTO RETRAIN DEMO - Webhook Style"
echo "=============================================="
echo ""

# ============================================
# BƯỚC 1: Cấu hình GitHub Secrets
# ============================================
echo ">>> BƯỚC 1: Cấu hình GitHub Secrets"
echo "-----------------------------------"
echo "Đặt các secrets này trong GitHub Settings:"
echo "  1. MLFLOW_TRACKING_URI - URL MLflow server"
echo "  2. AWS_ACCESS_KEY_ID - AWS access key"
echo "  3. AWS_SECRET_ACCESS_KEY - AWS secret key"
echo "  4. AWS_DEFAULT_REGION - Region (vd: us-east-1)"
echo "  5. S3_BUCKET_NAME - Bucket: retrain-data-fraud-detection"
echo ""
read -p "Đã cấu hình secrets? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Vui lòng cấu hình GitHub Secrets trước!"
    exit 1
fi

# ============================================
# BƯỚC 2: Start Infrastructure
# ============================================
echo ""
echo ">>> BƯỚC 2: Start Kafka + MLflow"
echo "-----------------------------------"
docker-compose -f docker-compose.mlops.yml up -d zookeeper kafka mlflow

export MLFLOW_TRACKING_URI="http://localhost:5000"
echo "MLflow UI: http://localhost:5000"

# ============================================
# BƯỚC 3: Start Producer + Consumer
# ============================================
echo ""
echo ">>> BƯỚC 3: Start Consumer (Terminal khác)"
echo "-----------------------------------"
echo "Mở terminal mới và chạy:"
echo "  python -m src.streaming.consumer"
echo ""
read -p "Đã start consumer? (y/n): " -n 1 -r
echo ""

echo ">>> BƯỚC 4: Start Producer (Terminal khác)"
echo "-----------------------------------"
echo "Mở terminal mới và chạy:"
echo "  python -m src.streaming.producer"
echo ""
read -p "Đã start producer? (y/n): " -n 1 -r
echo ""

# ============================================
# BƯỚC 5: Label Joiner - Gán nhãn
# ============================================
echo ""
echo ">>> BƯỚC 5: Label Joiner (gán nhãn từ dataset gốc)"
echo "-----------------------------------"
echo "Chạy label joiner để gán nhãn cho realtime data:"
echo ""
python -m src.pipeline.label_joiner

echo ""
echo "Đã gán nhãn xong. Data đã upload lên S3."
echo ""

# ============================================
# BƯỚC 6: Trigger Manual Retrain
# ============================================
echo ""
echo ">>> BƯỚC 6: Trigger Manual Retrain qua Webhook"
echo "-----------------------------------"
echo "Sử dụng GitHub CLI để trigger workflow:"
echo ""

REPO="minhquang1604/Financial_Fraud_Detection"
WEBHOOK_URL="https://api.github.com/repos/${REPO}/actions/workflows/retrain.yml/dispatch"

echo "Triggering GitHub Actions..."
curl -s -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer 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" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/${REPO}/actions/workflows/retrain.yml/dispatches \
  -d '{"ref":"main","inputs":{"trigger_type":"manual","ref_ratio":"0.75"}}'

echo ""
echo "=============================================="
echo "  WEBHOOK ĐÃ ĐƯỢC TRIGGER!"
echo "=============================================="
echo ""
echo "Kiểm tra tiến trình tại:"
echo "  https://github.com/${REPO}/actions"
echo ""
echo "Workflow sẽ:"
echo "  1. Download data từ S3"
echo "  2. Prepare mixed data (75% ref + 25% labeled)"
echo "  3. Run retrain pipeline với SMOTE"
echo "  4. Upload model mới lên S3"
echo "  5. Log metrics lên MLflow"
echo ""

# ============================================
# BƯỚC 7: Monitor Progress
# ============================================
echo ""
echo ">>> BƯỚC 7: Theo dõi tiến trình"
echo "-----------------------------------"
echo "Xem logs GitHub Actions:"
echo "  gh run watch"
echo ""
echo "Hoặc kiểm tra trực tiếp:"
echo "  https://github.com/${REPO}/actions/workflows/retrain.yml"
echo ""

# ============================================
# BƯỚC 8: Xem kết quả
# ============================================
echo ""
echo ">>> BƯỚC 8: Kiểm tra kết quả"
echo "-----------------------------------"
echo "Sau khi workflow hoàn tất:"
echo ""
echo "1. MLflow UI:"
echo "   http://localhost:5000"
echo ""
echo "2. Kiểm tra model mới:"
echo "   aws s3 ls s3://retrain-data-fraud-detection/model/"
echo ""
echo "3. Download model về local:"
echo "   aws s3 cp s3://retrain-data-fraud-detection/model/ ./model/ --recursive"
echo ""

echo "=============================================="
echo "  DEMO HOÀN TẤT!"
echo "=============================================="