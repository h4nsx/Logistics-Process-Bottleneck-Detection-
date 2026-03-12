"""
ML inference service: loads pre-trained IsolationForest models and scores
logistics cases against them.

Each process has its own model directory containing:
  model.pkl           - IsolationForest (sklearn)
  scaler.pkl          - StandardScaler fitted on training features
  baselines.json      - per-step mean / std / p95 / p99
  feature_schema.json - ordered feature column names
  score_quantiles.json - 100-point percentile lookup for risk mapping

Scoring pipeline:
  1. Build feature vector from step durations
  2. Scale with scaler.pkl
  3. decision_function() → raw anomaly score (lower = more anomalous)
  4. Map raw score → risk_percentile [0, 100] via quantile lookup
  5. Flag individual steps exceeding p95 or z-score ≥ 2
"""
import json
import logging
import os
from dataclasses import dataclass, field

import joblib
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_PROCESSES = [
    "TRUCKING_DELIVERY_FLOW",
    "IMPORT_CUSTOMS_CLEARANCE",
    "WAREHOUSE_FULFILLMENT",
]


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class StepAnomaly:
    step_code: str
    duration_min: float
    baseline_mean: float
    baseline_p95: float
    z_score: float | None
    exceeds_p95: bool
    is_anomaly: bool


@dataclass
class CasePrediction:
    case_id: str
    process_code: str
    anomaly_score: float
    risk_percentile: float
    is_anomaly: bool
    step_anomalies: list[StepAnomaly] = field(default_factory=list)
    total_duration_min: float = 0.0
    step_count: int = 0


# ── Model store ────────────────────────────────────────────────────────────────

class MLModelStore:
    """Loads and caches all process models from disk at startup."""

    def __init__(self, model_base_dir: str) -> None:
        self._base = model_base_dir
        self._artifacts: dict[str, dict] = {}
        self._load_all()

    def _load_one(self, process_code: str) -> dict:
        d = os.path.join(self._base, process_code)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Model directory not found: {d}")

        with open(os.path.join(d, "baselines.json"), encoding="utf-8") as f:
            baselines = json.load(f)
        with open(os.path.join(d, "feature_schema.json"), encoding="utf-8") as f:
            feature_schema = json.load(f)
        with open(os.path.join(d, "score_quantiles.json"), encoding="utf-8") as f:
            quantiles_data = json.load(f)

        model = joblib.load(os.path.join(d, "model.pkl"))
        scaler = joblib.load(os.path.join(d, "scaler.pkl"))

        return {
            "model": model,
            "scaler": scaler,
            "baselines": baselines,
            "step_codes": feature_schema["step_codes"],
            "step_feature_cols": feature_schema["step_feature_cols"],
            "all_feature_cols": feature_schema["all_feature_cols"],
            "quantiles": np.array(quantiles_data["raw_anomaly_quantiles"], dtype=float),
        }

    def _load_all(self) -> None:
        for pc in SUPPORTED_PROCESSES:
            try:
                self._artifacts[pc] = self._load_one(pc)
                logger.info("ML model loaded: %s", pc)
            except Exception as exc:
                logger.warning("Could not load ML model %s: %s", pc, exc)

    def get(self, process_code: str) -> dict | None:
        return self._artifacts.get(process_code)

    @property
    def loaded_processes(self) -> list[str]:
        return list(self._artifacts.keys())


# Singleton — initialized once in main.py lifespan
_store: MLModelStore | None = None


def init_models(model_base_dir: str) -> None:
    global _store
    _store = MLModelStore(model_base_dir)
    logger.info(
        "ML models ready: %s",
        _store.loaded_processes if _store else "none",
    )


def get_store() -> MLModelStore | None:
    return _store


# ── Feature engineering ────────────────────────────────────────────────────────

def _build_feature_vector(
    step_durations: dict[str, float],
    all_feature_cols: list[str],
    step_feature_cols: list[str],
) -> np.ndarray:
    """
    Build a 1-row feature matrix.

    step_durations: { step_code → duration_min }
    Missing steps are filled with 0 (same as training convention).
    """
    dur_by_col = {f"{s}_duration_min": d for s, d in step_durations.items()}
    durations = list(step_durations.values())

    values: list[float] = []
    for col in all_feature_cols:
        if col in dur_by_col:
            values.append(dur_by_col[col])
        elif col == "total_process_time_min":
            values.append(sum(durations))
        elif col == "max_step_duration_min":
            values.append(max(durations) if durations else 0.0)
        elif col == "mean_step_duration_min":
            values.append(sum(durations) / len(durations) if durations else 0.0)
        elif col == "std_step_duration_min":
            if len(durations) > 1:
                mu = sum(durations) / len(durations)
                values.append((sum((v - mu) ** 2 for v in durations) / len(durations)) ** 0.5)
            else:
                values.append(0.0)
        elif col == "step_count_present":
            values.append(float(len(step_durations)))
        elif col == "missing_step_count":
            values.append(float(len(step_feature_cols) - len(step_durations)))
        elif col == "missing_step_flag":
            values.append(1.0 if len(step_durations) < len(step_feature_cols) else 0.0)
        elif col in ("repeated_step_total", "repeated_step_flag"):
            # Repeated-step detection not done in online scoring
            values.append(0.0)
        else:
            values.append(0.0)

    return np.array(values, dtype=float).reshape(1, -1)


# ── Risk mapping ───────────────────────────────────────────────────────────────

def _raw_score_to_risk_percentile(raw_score: float, quantiles: np.ndarray) -> float:
    """
    Map IsolationForest decision_function score → risk percentile [0, 100].

    decision_function returns: lower = more anomalous.
    quantiles is the sorted array of training-set scores (1st..100th percentile).

    A score below the 1st training percentile → risk ≈ 100 (maximally anomalous).
    A score above the 100th training percentile → risk ≈ 0 (perfectly normal).
    """
    idx = int(np.searchsorted(quantiles, raw_score))
    normality = idx / len(quantiles)          # 0 = most anomalous, 1 = most normal
    return round((1.0 - normality) * 100.0, 1)


# ── Step-level anomaly detection ───────────────────────────────────────────────

def _detect_step_anomalies(
    step_durations: dict[str, float],
    baselines: dict,
) -> list[StepAnomaly]:
    """Flag steps that exceed p95 or have z-score ≥ 2."""
    step_baselines: dict[str, dict] = baselines.get("steps", {})
    results: list[StepAnomaly] = []

    for step_code, duration in step_durations.items():
        bl = step_baselines.get(step_code)
        if bl is None:
            continue

        mean = bl["mean"]
        std = bl["std"]
        p95 = bl["p95"]

        z_score = (duration - mean) / std if std > 0 else None
        exceeds_p95 = duration > p95
        is_anomaly = exceeds_p95 or (z_score is not None and z_score >= 2.0)

        results.append(
            StepAnomaly(
                step_code=step_code,
                duration_min=round(duration, 2),
                baseline_mean=round(mean, 2),
                baseline_p95=round(p95, 2),
                z_score=round(z_score, 3) if z_score is not None else None,
                exceeds_p95=exceeds_p95,
                is_anomaly=is_anomaly,
            )
        )

    # Return only anomalous steps, sorted by severity (deviation from p95)
    anomalous = [s for s in results if s.is_anomaly]
    return sorted(anomalous, key=lambda s: s.duration_min / s.baseline_p95, reverse=True)


# ── Public scoring API ─────────────────────────────────────────────────────────

def score_case(
    case_id: str,
    process_code: str,
    step_durations: dict[str, float],
) -> CasePrediction | None:
    """
    Score a single logistics case.

    Returns None if the process_code has no loaded model.
    Raises RuntimeError if init_models() was never called.
    """
    if _store is None:
        raise RuntimeError("ML models not initialized. Call init_models() first.")

    artifacts = _store.get(process_code)
    if artifacts is None:
        return None

    feature_vec = _build_feature_vector(
        step_durations,
        artifacts["all_feature_cols"],
        artifacts["step_feature_cols"],
    )

    scaled = artifacts["scaler"].transform(feature_vec)
    raw_score = float(artifacts["model"].decision_function(scaled)[0])
    risk_percentile = _raw_score_to_risk_percentile(raw_score, artifacts["quantiles"])

    # Case flagged as anomaly when risk > 50th percentile of training distribution
    is_anomaly = risk_percentile >= 50.0

    step_anomalies = _detect_step_anomalies(step_durations, artifacts["baselines"])
    total_duration = sum(step_durations.values())

    return CasePrediction(
        case_id=case_id,
        process_code=process_code,
        anomaly_score=round(raw_score, 6),
        risk_percentile=risk_percentile,
        is_anomaly=is_anomaly,
        step_anomalies=step_anomalies,
        total_duration_min=round(total_duration, 2),
        step_count=len(step_durations),
    )
