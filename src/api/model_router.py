import os
import sys
import time
import logging
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from src.api.model_loader import predict_proba_safe
from src.api.model_registry import BlueGreenRegistry

logger = logging.getLogger(__name__)


class ModelRouter:
    def __init__(self, registry: BlueGreenRegistry):
        self._registry = registry
        self._lock = threading.RLock()
        self._error_counts = {}
        self._latency_buffer = []

    def predict(self, X: pd.DataFrame) -> tuple:
        slot = self._registry.get_active_model()
        if slot.model is None or slot.status != "active":
            raise RuntimeError("No active model available")

        t0 = time.perf_counter()
        try:
            probs = predict_proba_safe(slot.model_data, X)
            prob = float(np.clip(probs[0], 0.0, 1.0))
            pred = int(prob >= slot.threshold)
            elapsed = time.perf_counter() - t0

            self._track_latency(elapsed)
            return prob, pred, slot.version, slot.run_id, elapsed
        except Exception as e:
            elapsed = time.perf_counter() - t0
            with self._lock:
                ver = slot.version
                self._error_counts[ver] = self._error_counts.get(ver, 0) + 1
            logger.error(f"Prediction error with model v{slot.version}: {e}")
            raise

    def _track_latency(self, latency: float):
        self._latency_buffer.append(latency)
        if len(self._latency_buffer) > 1000:
            self._latency_buffer = self._latency_buffer[-1000:]

    def get_error_count(self, version: str) -> int:
        return self._error_counts.get(version, 0)

    def get_recent_latency_stats(self) -> dict:
        buf = self._latency_buffer
        if not buf:
            return {"p50": 0, "p95": 0, "p99": 0, "avg": 0}
        arr = np.array(buf[-200:])
        return {
            "avg": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def get_version_info(self) -> dict:
        info = self._registry.get_active_info()
        latency = self.get_recent_latency_stats()
        info["latency"] = latency
        info["error_count"] = self.get_error_count(info["version"])
        return info
