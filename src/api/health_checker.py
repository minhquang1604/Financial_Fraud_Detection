import os
import sys
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import xgboost as xgb
from src.api.model_loader import predict_proba_safe
from src.api.model_registry import BlueGreenRegistry, ModelSlot

logger = logging.getLogger(__name__)


@dataclass
class HealthReport:
    passed: bool = False
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    active_f1: Optional[float] = None
    p95_latency: float = 0.0
    latency_ok: bool = False
    nan_count: int = 0
    nan_ok: bool = True
    f1_ok: bool = False
    warmup_ok: bool = False
    message: str = ""


class HealthChecker:
    WARMUP_ITERATIONS: int = 500
    F1_TOLERANCE: float = 0.05
    LATENCY_P99_MAX_MS: float = 500.0

    def __init__(self, registry: BlueGreenRegistry, validation_path: Optional[str] = None):
        self._registry = registry
        self._validation_data: Optional[pd.DataFrame] = None
        self._validation_labels: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        if validation_path and os.path.exists(validation_path):
            self._load_validation_data(validation_path)

    def _load_validation_data(self, path: str):
        try:
            df = pd.read_parquet(path)
            if "Class" in df.columns:
                self._validation_labels = df["Class"].values
                self._validation_data = df.drop(columns=["Class"])
            else:
                self._validation_data = df
                self._validation_labels = None
            logger.info(f"Loaded validation data: {len(df)} samples from {path}")
        except Exception as e:
            logger.warning(f"Could not load validation data from {path}: {e}")

    def set_validation_data(self, df: pd.DataFrame, labels: Optional[np.ndarray] = None):
        self._validation_data = df
        self._validation_labels = labels

    def warmup(self, slot: ModelSlot) -> bool:
        if slot.model is None:
            logger.error("Cannot warmup: model is None")
            return False

        try:
            n_features = len(slot.features) if slot.features else 34
            dummy = np.random.randn(self.WARMUP_ITERATIONS, n_features).astype(np.float32)
            dummy_df = pd.DataFrame(dummy, columns=slot.features if slot.features else None)

            for _ in range(3):
                predict_proba_safe(slot.model_data, dummy_df)

            latencies = []
            for _ in range(100):
                t0 = time.perf_counter()
                predict_proba_safe(slot.model_data, dummy_df)
                latencies.append((time.perf_counter() - t0) * 1000)

            p95 = float(np.percentile(latencies, 95))
            logger.info(
                f"Warmup complete for v{slot.version}: "
                f"p95_latency={p95:.2f}ms, "
                f"samples={self.WARMUP_ITERATIONS}"
            )
            return True
        except Exception as e:
            logger.error(f"Warmup failed for v{slot.version}: {e}")
            return False

    def validate_candidate(self, candidate: ModelSlot) -> HealthReport:
        report = HealthReport()
        active_slot = self._registry.get_active_model()

        if candidate.model is None:
            report.message = "Candidate model is None"
            return report

        report.warmup_ok = self.warmup(candidate)
        if not report.warmup_ok:
            report.message = "Warmup failed"
            return report

        if self._validation_data is not None:
            try:
                n_features = len(candidate.features) if candidate.features else 34
                val_df = self._validation_data
                if val_df.shape[1] > n_features:
                    val_df = val_df.iloc[:, :n_features]
                elif val_df.shape[1] < n_features:
                    pad = np.zeros((len(val_df), n_features - val_df.shape[1]))
                    val_df = pd.concat(
                        [val_df, pd.DataFrame(pad, columns=candidate.features[-pad.shape[1]:])],
                        axis=1
                    )

                val_df.columns = candidate.features[:val_df.shape[1]]

                t0 = time.perf_counter()
                probs = predict_proba_safe(candidate.model_data, val_df)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                report.p95_latency = elapsed_ms
                report.latency_ok = elapsed_ms < self.LATENCY_P99_MAX_MS

                preds = (probs >= candidate.threshold).astype(int)
                nan_mask = np.isnan(probs) | np.isinf(probs)
                report.nan_count = int(nan_mask.sum())
                report.nan_ok = report.nan_count == 0

                if self._validation_labels is not None:
                    y_true = self._validation_labels[:len(preds)]
                    tp = int(((preds == 1) & (y_true == 1)).sum())
                    fp = int(((preds == 1) & (y_true == 0)).sum())
                    fn = int(((preds == 0) & (y_true == 1)).sum())
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    report.f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    report.precision = precision
                    report.recall = recall

                    active_f1 = active_slot.metrics.get("F1", 0.0)
                    report.active_f1 = active_f1
                    if active_f1 > 0:
                        report.f1_ok = report.f1_score >= (active_f1 - self.F1_TOLERANCE)
                    else:
                        report.f1_ok = True

                logger.info(
                    f"Validation for v{candidate.version}: "
                    f"F1={report.f1_score:.4f}, "
                    f"latency_p95={report.p95_latency:.2f}ms, "
                    f"nan={report.nan_count}"
                )
            except Exception as e:
                logger.error(f"Validation failed for v{candidate.version}: {e}")
                report.message = f"Validation error: {e}"
                return report
        else:
            report.f1_ok = True
            report.latency_ok = True
            report.nan_ok = True
            report.message = "No validation data; skipping metric comparison"

        report.passed = report.latency_ok and report.nan_ok and report.f1_ok
        if not report.message:
            report.message = (
                "All checks passed" if report.passed
                else f"Checks failed: latency_ok={report.latency_ok}, nan_ok={report.nan_ok}, f1_ok={report.f1_ok}"
            )
        return report
