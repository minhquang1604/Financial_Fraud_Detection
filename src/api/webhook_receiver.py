import os
import sys
import logging
import requests
from datetime import datetime
from typing import Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GITHUB_REPO = os.environ.get("GITHUB_REPO", "owner/repo")
GITHUB_WORKFLOW = "retrain.yml"
PAT_TOKEN = os.environ.get("PAT_TOKEN", "")

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")


def trigger_github_workflow(event_type: str = "drift_alert") -> Dict[str, Any]:
    if not PAT_TOKEN:
        logger.warning("PAT_TOKEN not set, skipping GitHub Actions trigger")
        return {"success": False, "error": "PAT_TOKEN not set"}
    
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{GITHUB_WORKFLOW}/dispatches"
    
    headers = {
        "Authorization": f"Bearer {PAT_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    payload = {
        "ref": "main",
        "inputs": {
            "trigger_type": event_type
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code in [200, 201]:
            logger.info(f"GitHub Actions triggered successfully: {event_type}")
            return {"success": True, "status_code": response.status_code}
        else:
            logger.error(f"GitHub Actions trigger failed: {response.status_code} - {response.text}")
            return {"success": False, "error": response.text}
            
    except Exception as e:
        logger.error(f"Error triggering GitHub Actions: {e}")
        return {"success": False, "error": str(e)}


def handle_alertmanager_webhook(payload: Dict[str, Any]) -> Dict[str, Any]:
    alerts = payload.get("alerts", [])
    
    if not alerts:
        return {"success": False, "error": "No alerts in payload"}
    
    fired_alerts = [a for a in alerts if a.get("status") == "fired"]
    
    if not fired_alerts:
        return {"success": False, "error": "No fired alerts"}
    
    logger.info(f"Received {len(fired_alerts)} fired alerts")
    
    alert_names = [a.get("labels", {}).get("alertname", "unknown") for a in fired_alerts]
    logger.info(f"Alert names: {alert_names}")
    
    if any("Drift" in name for name in alert_names):
        return trigger_github_workflow(event_type="drift_alert")
    
    return {"success": False, "error": "No drift-related alerts"}


def handle_drift_webhook(payload: Dict[str, Any]) -> Dict[str, Any]:
    drift_detected = payload.get("drift_detected", False)
    
    if drift_detected:
        logger.info("Drift detected, triggering retrain...")
        return trigger_github_workflow(event_type="drift_alert")
    
    return {"success": False, "error": "No drift detected"}


if __name__ == "__main__":
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    @app.route("/webhook", methods=["POST"])
    def webhook():
        payload = request.json
        
        logger.info(f"Received webhook: {payload}")
        
        result = handle_alertmanager_webhook(payload)
        
        return jsonify(result)
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy"})
    
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)