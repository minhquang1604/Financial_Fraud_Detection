import os
import sys
import time
import logging
import threading
from collections import defaultdict, deque
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.api.model_registry import BlueGreenRegistry

logger = logging.getLogger(__name__)


class RollbackManager:
    ERROR_RATE_THRESHOLD: float = 0.1
    LATENCY_P99_THRESHOLD_MS: float = 1000.0
    OBSERVATION_WINDOW: int = 300
    MIN_OBSERVATIONS: int = 50

    def __init__(self, registry: BlueGreenRegistry):
        self._registry = registry
        self._lock = threading.RLock()
        self._predictions: dict = defaultdict(lambda: deque(maxlen=10000))
        self._errors: dict = defaultdict(lambda: deque(maxlen=10000))
        self._latencies: dict = defaultdict(lambda: deque(maxlen=10000))
        self._rollback_count = 0
        self._last_rollback_time = 0.0
        self._rollback_cooldown = 300.0

    def record_prediction(self, version: str, latency_ms: float, success: bool):
        now = time.time()
        self._predictions[version].append(now)
        self._latencies[version].append(latency_ms)
        if not success:
            self._errors[version].append(now)

    def check_rollback_needed(self) -> Optional[str]:
        active = self._registry.get_active_model()
        version = active.version

        if len(self._predictions[version]) < self.MIN_OBSERVATIONS:
            return None

        now = time.time()
        cutoff = now - self.OBSERVATION_WINDOW

        with self._lock:
            recent_preds = [t for t in self._predictions[version] if t > cutoff]
            recent_errors = [t for t in self._errors[version] if t > cutoff]
            recent_lats = [
                lat for t, lat in zip(self._predictions[version], self._latencies[version])
                if t > cutoff
            ]

        n_preds = len(recent_preds)
        if n_preds < self.MIN_OBSERVATIONS:
            return None

        error_rate = len(recent_errors) / max(n_preds, 1)
        p99_latency = sorted(recent_lats)[int(len(recent_lats) * 0.99)] if recent_lats else 0

        reasons = []
        if error_rate > self.ERROR_RATE_THRESHOLD:
            reasons.append(f"error_rate={error_rate:.3f} > {self.ERROR_RATE_THRESHOLD}")
        if p99_latency > self.LATENCY_P99_THRESHOLD_MS:
            reasons.append(f"p99_latency={p99_latency:.1f}ms > {self.LATENCY_P99_THRESHOLD_MS}ms")

        if reasons and (now - self._last_rollback_time) > self._rollback_cooldown:
            logger.warning(f"Rollback triggered for v{version}: {', '.join(reasons)}")
            return self._get_previous_version(version)

        return None

    def execute_rollback(self) -> bool:
        now = time.time()
        if (now - self._last_rollback_time) < self._rollback_cooldown:
            logger.warning("Rollback cooldown active, skipping")
            return False

        active = self._registry.get_active_model()
        current_version = active.version

        if len(self._registry._history) < 2:
            logger.warning("No previous version to rollback to")
            return False

        if self._registry.swap():
            self._rollback_count += 1
            self._last_rollback_time = now
            new_active = self._registry.get_active_model()
            logger.info(
                f"Rollback executed: v{current_version} \u2192 v{new_active.version} "
                f"(total rollbacks: {self._rollback_count})"
            )
            return True

        return False

    def _get_previous_version(self, current_version: str) -> Optional[str]:
        with self._lock:
            if hasattr(self._registry, '_history') and len(self._registry._history) >= 2:
                for entry in reversed(self._registry._history[:-1]):
                    if entry.get("from_version") and entry["from_version"] != current_version:
                        return entry["from_version"]
        return None

    @property
    def stats(self) -> dict:
        return {
            "rollback_count": self._rollback_count,
            "last_rollback_time": self._last_rollback_time,
        }
