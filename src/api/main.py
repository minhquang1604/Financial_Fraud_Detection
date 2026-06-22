import os
import sys
import time
import logging
import threading

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from src.train.utils import engineer_features, get_feature_columns
from src.api.schemas import PredictionRequest, PredictionResponse
from src.api.model_loader import load_model_from_mlflow
from src.api.model_registry import BlueGreenRegistry
from src.api.model_router import ModelRouter
from src.api.health_checker import HealthChecker
from src.api.rollback_manager import RollbackManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREDICTIONS_TOTAL = Counter(
    'fraud_predictions_total', 'Total predictions by class',
    ['prediction']
)
INFERENCE_LATENCY = Histogram(
    'inference_latency_seconds', 'Inference latency in seconds',
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)
FRAUD_PROBABILITY = Gauge(
    'fraud_prediction_probability', 'Current fraud prediction probability'
)
MODEL_SWITCHES = Counter(
    'model_switches_total', 'Total model switch events'
)
MODEL_ROLLBACKS = Counter(
    'model_rollbacks_total', 'Total model rollback events'
)
MODEL_ACTIVE_VERSION = Gauge(
    'model_active_version', 'Active model version number'
)
MODEL_ACTIVE_SWITCH_TIMESTAMP = Gauge(
    'model_active_switch_timestamp_seconds', 'Timestamp of last model switch'
)

registry = BlueGreenRegistry()
router = None
health_checker = None
rollback_manager = None
watcher = None


class RunningStats:
    def __init__(self, window_size: int = 10000):
        self.window_size = window_size
        self.amounts = []
        self.mean_amt = 0.0
        self.median_amt = 0.0
        self.threshold_95 = 0.0
        self.alpha = 0.01

    def update(self, amount: float):
        self.amounts.append(amount)
        if len(self.amounts) > self.window_size:
            self.amounts.pop(0)
        if len(self.amounts) >= 100:
            self.mean_amt = np.mean(self.amounts)
            self.median_amt = np.median(self.amounts)
            self.threshold_95 = np.percentile(self.amounts, 95)

    def get_stats(self) -> dict:
        return {
            "mean_amt": self.mean_amt if self.mean_amt > 0 else 1.0,
            "median_amt": self.median_amt if self.median_amt > 0 else 1.0,
            "threshold_95": self.threshold_95 if self.threshold_95 > 0 else 1000.0
        }


running_stats = RunningStats()


MODEL_UPDATE_SECRET = os.environ.get("MODEL_UPDATE_SECRET", "")


class ModelUpdateWatcher:
    def __init__(self, registry: BlueGreenRegistry, checker: HealthChecker,
                 rollback: RollbackManager, poll_interval: int = 60):
        self._registry = registry
        self._health_checker = checker
        self._rollback_manager = rollback
        self._poll_interval = poll_interval
        self._running = False
        self._thread = None
        self._loaded_versions = set()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True,
                                        name="model-watcher")
        self._thread.start()
        logger.info(f"ModelUpdateWatcher started (poll every {self._poll_interval}s)")

    def stop(self):
        self._running = False
        logger.info("ModelUpdateWatcher stopped")

    def _poll_loop(self):
        while self._running:
            try:
                self._check_for_updates()
            except Exception as e:
                logger.error(f"Model update check failed: {e}")
            time.sleep(self._poll_interval)

    def _check_for_updates(self):
        active = self._registry.get_active_model()
        self._loaded_versions.add(active.version)

        try:
            import mlflow
            client = mlflow.tracking.MlflowClient()
            all_versions = client.search_model_versions(
                f"name='FraudDetectionModel'"
            )
            staging_versions = [
                v for v in all_versions
                if v.current_stage in ("Staging", "Production")
            ]
            latest = max(staging_versions, key=lambda v: int(v.version))
            if latest.version not in self._loaded_versions:
                logger.info(
                    f"New model version detected: v{latest.version} "
                    f"(stage={latest.current_stage})"
                )
                self._load_and_validate(latest.version)
        except Exception as e:
            logger.warning(f"MLflow poll check failed: {e}")

    def load_and_validate(self, version: str):
        self._load_and_validate(version)

    def _load_and_validate(self, version: str):
        ok = self._registry.load_standby(version)
        if not ok:
            return False

        candidate = self._registry.get_standby_slot()
        report = self._health_checker.validate_candidate(candidate)

        if report.passed:
            if self._registry.swap():
                MODEL_SWITCHES.inc()
                MODEL_ACTIVE_VERSION.set(float(version))
                MODEL_ACTIVE_SWITCH_TIMESTAMP.set(time.time())
                self._loaded_versions.add(version)
                logger.info(
                    f"Successfully switched to model v{version}"
                )
                return True
        else:
            logger.warning(
                f"Model v{version} failed health check: {report.message}"
            )
            candidate.status = "failed"
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router, health_checker, rollback_manager, watcher
    for attempt in range(1, 13):
        try:
            registry.initialize()
            break
        except Exception as e:
            logger.warning(f"Model init attempt {attempt}/12 failed: {e}")
            if attempt < 12:
                time.sleep(10)
    router = ModelRouter(registry)
    health_checker = HealthChecker(
        registry=registry,
        validation_path=os.environ.get("VALIDATION_DATA_PATH")
    )
    rollback_manager = RollbackManager(registry)
    watcher = ModelUpdateWatcher(registry, health_checker, rollback_manager,
                                 poll_interval=int(os.environ.get("MODEL_POLL_INTERVAL", "60")))
    watcher.start()
    if registry.is_initialized:
        info = registry.get_active_info()
        MODEL_ACTIVE_VERSION.set(float(info["version"]))
        MODEL_ACTIVE_SWITCH_TIMESTAMP.set(time.time())
        logger.info(
            f"Blue-Green initialized. Active: {info['active_color']} "
            f"v{info['version']} (F1={info['metrics'].get('F1', 'N/A')})"
        )
    else:
        logger.warning("Model not loaded at startup — watcher will retry")
    yield
    if watcher:
        watcher.stop()


app = FastAPI(
    title="Fraud Detection API (Blue-Green)",
    description="Real-time Credit Card Fraud Detection with Blue-Green Model Deployment",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    if not registry.is_initialized:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = registry.get_active_info()
    standby = registry.get_standby_info()
    return {
        "status": "healthy",
        "model_loaded": True,
        "active_color": info["active_color"],
        "active_model": {
            "version": info["version"],
            "run_id": info["run_id"],
            "status": info["status"],
            "loaded_at": info["loaded_at"],
            "f1_score": info["metrics"].get("F1", "N/A"),
        },
        "standby_model": {
            "version": standby["version"] or "none",
            "status": standby["status"],
        },
    }


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not registry.is_initialized:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        raw_features = request.features.model_dump()
        df = pd.DataFrame([raw_features])

        amount = raw_features.get("Amount", 0)
        running_stats.update(amount)
        stats = running_stats.get_stats()
        fake_ref_df = pd.DataFrame({
            "Amount": [stats["mean_amt"], stats["median_amt"], stats["threshold_95"]]
        })

        df = engineer_features(df, reference_df=fake_ref_df)
        feature_cols = get_feature_columns()
        X = df[feature_cols]

        prob, pred, model_version, model_run_id, latency = router.predict(X)

        PREDICTIONS_TOTAL.labels(prediction=str(pred)).inc()
        FRAUD_PROBABILITY.set(float(prob))

        if rollback_manager:
            rollback_manager.record_prediction(
                version=model_version,
                latency_ms=latency * 1000,
                success=True
            )

        return PredictionResponse(
            transaction_time=raw_features['Time'],
            fraud_probability=float(prob),
            prediction=pred,
            model_version=model_version,
            model_run_id=model_run_id,
            message="🚨 FRAUD DETECTED" if pred == 1 else "✅ Normal transaction"
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Predict error: {e}")
        logger.error(traceback.format_exc())
        if router and rollback_manager:
            try:
                info = registry.get_active_info()
                rollback_manager.record_prediction(
                    version=info["version"],
                    latency_ms=(time.time() - start) * 1000,
                    success=False
                )
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        INFERENCE_LATENCY.observe(time.time() - start)


@app.get("/model/info")
async def model_info():
    return {
        "active": registry.get_active_info(),
        "standby": registry.get_standby_info(),
        "rollback": rollback_manager.stats if rollback_manager else {},
    }


@app.get("/model/history")
async def model_history(limit: int = 10):
    return {"history": registry.get_history(limit=limit)}


@app.post("/model/rollback")
async def model_rollback():
    if not rollback_manager:
        raise HTTPException(status_code=503, detail="Rollback manager not available")
    if rollback_manager.execute_rollback():
        MODEL_ROLLBACKS.inc()
        info = registry.get_active_info()
        return {
            "status": "rolled_back",
            "active_version": info["version"],
            "active_color": info["active_color"],
        }
    raise HTTPException(status_code=400, detail="Rollback failed or cooldown active")


@app.post("/model/update")
async def model_update(version: str, secret: str = ""):
    if MODEL_UPDATE_SECRET and secret != MODEL_UPDATE_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    if not watcher:
        raise HTTPException(status_code=503, detail="Update watcher not available")

    logger.info(f"Manual model update requested: v{version}")
    ok = watcher.load_and_validate(version)
    if ok:
        info = registry.get_active_info()
        return {
            "status": "switched",
            "active_version": info["version"],
            "active_color": info["active_color"],
        }
    info = registry.get_active_info()
    standby = registry.get_standby_info()
    return {
        "status": "failed",
        "active_version": info["version"],
        "standby_version": standby["version"],
        "standby_status": standby["status"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
