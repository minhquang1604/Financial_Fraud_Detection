import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from prometheus_client.core import CollectorRegistry, REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PrometheusMetricsExporter:
    def __init__(
        self,
        port: int = 8001,
        registry: Optional[CollectorRegistry] = None
    ):
        self.port = port
        self.registry = registry or REGISTRY
        
        self._setup_metrics()
    
    def _setup_metrics(self):
        self.prediction_count = Counter(
            'fraud_predictions_total',
            'Total number of predictions',
            ['prediction']
        )
        
        self.predictionprobability = Histogram(
            'fraud_prediction_probability',
            'Fraud prediction probability',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        self.drift_score = Gauge(
            'data_drift_score',
            'Current data drift score'
        )
        
        self.concept_drift_score = Gauge(
            'concept_drift_score',
            'Current concept drift score'
        )
        
        self.model_f1 = Gauge(
            'model_f1_score',
            'Current model F1 score'
        )
        
        self.model_precision = Gauge(
            'model_precision',
            'Current model precision'
        )
        
        self.model_recall = Gauge(
            'model_recall',
            'Current model recall'
        )
        
        self.drift_alert = Gauge(
            'drift_alert_triggered',
            'Whether drift alert is triggered (1) or not (0)'
        )
        
        self.training_data_count = Gauge(
            'training_data_count',
            'Number of samples in training data'
        )
        
        self.inference_latency = Histogram(
            'inference_latency_seconds',
            'Inference latency in seconds'
        )
    
    def export_prediction_metrics(self, probability: float, prediction: int):
        self.prediction_count.labels(prediction=str(prediction)).inc()
        self.predictionprobability.observe(probability)
        logger.debug(f"Exported prediction: prob={probability}, pred={prediction}")
    
    def export_drift_metrics(
        self,
        data_drift_score: float,
        concept_drift_score: float,
        alert_triggered: bool
    ):
        self.drift_score.set(data_drift_score)
        self.concept_drift_score.set(concept_drift_score)
        self.drift_alert.set(1 if alert_triggered else 0)
        
        logger.info(
            f"Drift metrics - data: {data_drift_score:.4f}, "
            f"concept: {concept_drift_score:.4f}, alert: {alert_triggered}"
        )
    
    def export_model_metrics(
        self,
        f1: float,
        precision: float,
        recall: float
    ):
        self.model_f1.set(f1)
        self.model_precision.set(precision)
        self.model_recall.set(recall)
        
        logger.info(f"Model metrics - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    def export_data_stats(self, data_count: int):
        self.training_data_count.set(data_count)
        
        logger.info(f"Training data count: {data_count}")
    
    def export_inference_latency(self, latency: float):
        self.inference_latency.observe(latency)
    
    def start_server(self):
        logger.info(f"Starting Prometheus metrics server on port {self.port}")
        start_http_server(self.port, registry=self.registry)


class MetricsCollector:
    def __init__(self):
        self.predictions = []
        self.prediction_times = []
    
    def collect_prediction(
        self,
        transaction_id: str,
        probability: float,
        prediction: int,
        actual_label: Optional[int] = None,
        inference_time: float = 0.0
    ):
        self.predictions.append({
            "transaction_id": transaction_id,
            "probability": probability,
            "prediction": prediction,
            "actual_label": actual_label,
            "inference_time": inference_time,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_batch_metrics(self, window_size: int = 1000) -> Dict[str, Any]:
        if not self.predictions:
            return {}
        
        recent = self.predictions[-window_size:]
        
        probabilities = [p["probability"] for p in recent]
        predictions = [p["prediction"] for p in recent]
        
        metrics = {
            "total_predictions": len(recent),
            "mean_probability": float(np.mean(probabilities)),
            "std_probability": float(np.std(probabilities)),
            "fraud_ratio": float(np.mean(predictions)),
            "avg_inference_time": float(np.mean([p["inference_time"] for p in recent])),
        }
        
        labels = [p["actual_label"] for p in recent if p["actual_label"] is not None]
        
        if labels and len(labels) > 0:
            from sklearn.metrics import (
                precision_score, 
                recall_score, 
                f1_score,
                confusion_matrix
            )
            
            pred_labels = [p["prediction"] for p in recent if p["actual_label"] is not None]
            
            metrics["precision"] = float(precision_score(labels, pred_labels, zero_division=0))
            metrics["recall"] = float(recall_score(labels, pred_labels, zero_division=0))
            metrics["f1"] = float(f1_score(labels, pred_labels, zero_division=0))
            
            tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()
            metrics["true_positives"] = int(tp)
            metrics["false_positives"] = int(fp)
            metrics["true_negatives"] = int(tn)
            metrics["false_negatives"] = int(fn)
        
        return metrics
    
    def clear(self):
        self.predictions.clear()