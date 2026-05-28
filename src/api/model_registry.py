import os
import sys
import threading
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.api.model_loader import load_model_from_mlflow

logger = logging.getLogger(__name__)


@dataclass
class ModelSlot:
    model: Optional[object] = None
    model_data: Optional[dict] = None
    threshold: float = 0.5
    features: list = field(default_factory=list)
    version: str = ""
    run_id: str = ""
    status: str = "empty"
    loaded_at: Optional[datetime] = None
    metrics: dict = field(default_factory=dict)


class BlueGreenRegistry:
    def __init__(self):
        self._blue = ModelSlot()
        self._green = ModelSlot()
        self._active_color = "blue"
        self._lock = threading.RLock()
        self._history = []

    def initialize(self, stage: str = "Production"):
        logger.info("Loading initial model...")
        model_data = load_model_from_mlflow(stage=stage)
        self._blue.model = model_data["model"]
        self._blue.model_data = model_data
        self._blue.threshold = float(model_data.get("threshold", 0.5))
        self._blue.features = model_data.get("features", [])
        self._blue.version = str(model_data.get("version", ""))
        self._blue.run_id = str(model_data.get("run_id", ""))
        self._blue.status = "active"
        self._blue.loaded_at = datetime.now()
        self._blue.metrics = model_data.get("metrics", {})
        self._active_color = "blue"
        self._history.append({
            "version": self._blue.version,
            "run_id": self._blue.run_id,
            "color": "blue",
            "action": "initial_load",
            "timestamp": datetime.now().isoformat()
        })
        logger.info(f"Initial model loaded: v{self._blue.version} (run: {self._blue.run_id})")

    def get_active_model(self) -> ModelSlot:
        return self._blue if self._active_color == "blue" else self._green

    def get_standby_slot(self) -> ModelSlot:
        return self._green if self._active_color == "blue" else self._blue

    def load_standby(self, version: str) -> bool:
        try:
            slot = self.get_standby_slot()
            slot.status = "loading"
            logger.info(f"Loading model v{version} into standby slot...")
            model_data = load_model_from_mlflow(version=int(version))
            slot.model = model_data["model"]
            slot.model_data = model_data
            slot.threshold = float(model_data.get("threshold", 0.5))
            slot.features = model_data.get("features", [])
            slot.version = str(version)
            slot.run_id = str(model_data.get("run_id", ""))
            slot.loaded_at = datetime.now()
            slot.metrics = model_data.get("metrics", {})
            slot.status = "standby"
            logger.info(f"Standby slot loaded: v{version} (run: {slot.run_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to load standby model v{version}: {e}")
            slot = self.get_standby_slot()
            slot.status = "failed"
            return False

    def swap(self) -> bool:
        with self._lock:
            old_color = self._active_color
            new_color = "green" if old_color == "blue" else "blue"
            old_slot = getattr(self, f"_{old_color}")
            new_slot = getattr(self, f"_{new_color}")

            if new_slot.status != "standby":
                logger.error(f"Cannot swap: standby slot status is '{new_slot.status}'")
                return False

            self._active_color = new_color
            old_slot.status = "standby"
            new_slot.status = "active"

            self._history.append({
                "version": new_slot.version,
                "run_id": new_slot.run_id,
                "from": old_color,
                "to": new_color,
                "from_version": old_slot.version,
                "timestamp": datetime.now().isoformat()
            })

            if len(self._history) > 50:
                self._history = self._history[-50:]

            logger.info(
                f"Swapped: {old_color} (v{old_slot.version}) "
                f"\u2192 {new_color} (v{new_slot.version})"
            )
            return True

    def get_active_info(self) -> dict:
        slot = self.get_active_model()
        return {
            "active_color": self._active_color,
            "version": slot.version,
            "run_id": slot.run_id,
            "threshold": slot.threshold,
            "features_count": len(slot.features),
            "status": slot.status,
            "loaded_at": slot.loaded_at.isoformat() if slot.loaded_at else None,
            "metrics": slot.metrics,
        }

    def get_standby_info(self) -> dict:
        slot = self.get_standby_slot()
        return {
            "version": slot.version,
            "run_id": slot.run_id,
            "threshold": slot.threshold,
            "status": slot.status,
            "loaded_at": slot.loaded_at.isoformat() if slot.loaded_at else None,
        }

    def get_history(self, limit: int = 10) -> list:
        return self._history[-limit:]

    @property
    def is_initialized(self) -> bool:
        slot = self.get_active_model()
        return slot.status == "active" and slot.model is not None
