from fastapi import FastAPI, HTTPException, UploadFile, File
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
import os
import io
from datetime import datetime
import numpy as np

# ===== Process AI imports (event-based bottleneck) =====
from api.process_ai.inference import load_process_artifacts
from api.process_ai.validate import validate_events_df
from api.process_ai.features import build_case_feature_matrix

# ======================================================
# Paths / Config
# ======================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
PROCESS_MODEL_ROOT = os.path.join(MODEL_DIR, "process_models")
REGISTRY_DIR = os.path.join(PROJECT_ROOT, "registry", "synth_optimal_v1")

# Legacy models (driver/fleet/ops)
models: Dict[str, Any] = {}

# Process artifacts cache
_process_artifacts_cache: Dict[str, Any] = {}

# process_code -> process_id
PROCESS_ID_MAP = {
    "TRUCKING_DELIVERY_FLOW": 1,
    "WAREHOUSE_FULFILLMENT": 2,
    "IMPORT_CUSTOMS_CLEARANCE": 3,
}


# ======================================================
# Lifespan: load legacy models on startup
# ======================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"📂 Loading legacy models from: {MODEL_DIR}")
    try:
        models["driver"] = joblib.load(os.path.join(MODEL_DIR, "driver_ai.pkl"))
        models["fleet"] = joblib.load(os.path.join(MODEL_DIR, "fleet_ai.pkl"))
        models["ops"] = joblib.load(os.path.join(MODEL_DIR, "ops_ai.pkl"))
        print("✅ Legacy driver/fleet/ops models loaded!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load legacy models (driver/fleet/ops). Error: {e}")
    yield


app = FastAPI(title="Logistics AI Core", lifespan=lifespan)


# ======================================================
# Legacy driver/fleet/ops (giữ để bổ trợ)
# ======================================================
class ShipmentInput(BaseModel):
    case_id: str

    # driver
    years_experience: int
    total_accidents: int
    avg_ontime_rate: float
    avg_miles_per_month: float
    avg_mpg_driver: float

    # fleet
    truck_age: int
    lifetime_maint_cost: float
    maint_frequency: int
    total_downtime: float
    avg_monthly_miles_truck: float

    # ops
    detention_hours: float
    real_mpg_trip: float
    delay_hours: float
    actual_distance_miles: float


def get_risk_probability(model, df_input: pd.DataFrame) -> float:
    """Return prob(class=1) if predict_proba exists, else 0."""
    try:
        probs = model.predict_proba(df_input)[0]
        if len(probs) > 1:
            return float(probs[1])
        return float(probs[0])
    except Exception:
        return 0.0


def process_single_shipment(item: ShipmentInput) -> Dict[str, Any]:
    """Legacy scoring: driver/fleet/ops + 3-step heuristic explainability demo."""
    if not models:
        raise HTTPException(status_code=500, detail="Legacy models not loaded")

    # driver
    driver_df = pd.DataFrame([{
        "years_experience": item.years_experience,
        "total_accidents": item.total_accidents,
        "avg_ontime_rate": item.avg_ontime_rate,
        "avg_miles_per_month": item.avg_miles_per_month,
        "avg_mpg": item.avg_mpg_driver,
    }])
    driver_prob = get_risk_probability(models["driver"], driver_df)

    # fleet
    fleet_df = pd.DataFrame([{
        "truck_age": item.truck_age,
        "lifetime_maint_cost": item.lifetime_maint_cost,
        "maint_frequency": item.maint_frequency,
        "total_downtime": item.total_downtime,
        "avg_monthly_miles": item.avg_monthly_miles_truck,
    }])
    fleet_prob = get_risk_probability(models["fleet"], fleet_df)

    # ops
    ops_df = pd.DataFrame([{
        "detention_hours": item.detention_hours,
        "real_mpg": item.real_mpg_trip,
        "delay_hours": item.delay_hours,
        "actual_distance_miles": item.actual_distance_miles,
    }])
    ops_prob = get_risk_probability(models["ops"], ops_df)

    contributors = []
    total_duration = 0.0

    # STEP_01_LOADING
    step1_actual = 60 + (item.detention_hours * 60)
    step1_p95 = 120.0
    if step1_actual > step1_p95:
        contributors.append({
            "step_code": "STEP_01_LOADING",
            "actual_duration_min": round(step1_actual, 1),
            "historical_p95_min": step1_p95,
            "deviation_factor": round(step1_actual / step1_p95, 2),
        })
    total_duration += step1_actual

    # STEP_02_TRANSIT
    est_transit_time = (item.actual_distance_miles / 50) * 60
    delay_factor = 1.0 + (fleet_prob * 0.5) + (driver_prob * 0.3)
    step2_actual = est_transit_time * delay_factor
    step2_p95 = est_transit_time * 1.2
    if step2_actual > step2_p95:
        contributors.append({
            "step_code": "STEP_02_TRANSIT",
            "actual_duration_min": round(step2_actual, 1),
            "historical_p95_min": round(step2_p95, 1),
            "deviation_factor": round(step2_actual / step2_p95, 2),
        })
    total_duration += step2_actual

    # STEP_03_UNLOADING
    step3_actual = 45 + (item.delay_hours * 30)
    step3_p95 = 90.0
    if step3_actual > step3_p95:
        contributors.append({
            "step_code": "STEP_03_UNLOADING",
            "actual_duration_min": round(step3_actual, 1),
            "historical_p95_min": step3_p95,
            "deviation_factor": round(step3_actual / step3_p95, 2),
        })
    total_duration += step3_actual

    contributors.sort(key=lambda x: x["deviation_factor"], reverse=True)

    total_risk_score = max(driver_prob, fleet_prob, ops_prob) * 100
    is_anomaly = len(contributors) > 0

    return {
        "meta": {
            "process_code": "LEGACY_DELIVERY_FLOW",
            "case_id": str(item.case_id),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "analysis": {"risk_score": round(total_risk_score, 1), "is_anomaly": bool(is_anomaly)},
        "explainability": {"contributors": contributors, "total_duration_min": round(total_duration, 1)},
    }


@app.post("/analyze_shipment")
def analyze_shipment(item: ShipmentInput):
    return process_single_shipment(item)


@app.post("/analyze_batch_csv")
async def analyze_batch_csv(file: UploadFile = File(...)):
    if not models:
        raise HTTPException(status_code=500, detail="Legacy models not loaded")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")

    total_cases = len(df)
    total_risk = 0.0
    anomaly_count = 0

    for i, row in df.iterrows():
        try:
            item = ShipmentInput(
                case_id=str(row.get("trip_id", f"ROW_{i}")),
                years_experience=int(row["years_experience"]),
                total_accidents=int(row["total_accidents"]),
                avg_ontime_rate=float(row["avg_ontime_rate"]),
                avg_miles_per_month=float(row["avg_miles_per_month"]),
                avg_mpg_driver=float(row["avg_mpg_driver"]),
                truck_age=int(row["truck_age"]),
                lifetime_maint_cost=float(row["lifetime_maint_cost"]),
                maint_frequency=int(row["maint_frequency"]),
                total_downtime=float(row["total_downtime"]),
                avg_monthly_miles_truck=float(row["avg_monthly_miles_truck"]),
                detention_hours=float(row["detention_hours"]),
                real_mpg_trip=float(row["real_mpg_trip"]),
                delay_hours=float(row["delay_hours"]),
                actual_distance_miles=float(row["actual_distance_miles"]),
            )
            res = process_single_shipment(item)
            total_risk += float(res["analysis"]["risk_score"])
            if res["analysis"]["is_anomaly"]:
                anomaly_count += 1
        except Exception:
            continue

    avg_risk = round(total_risk / max(1, total_cases), 2)
    anomaly_rate = round(anomaly_count / max(1, total_cases), 4)

    return {
        "total_cases": total_cases,
        "avg_risk": avg_risk,
        "anomaly_rate": anomaly_rate,
    }


# ======================================================
# Process Bottleneck (event-based V2)
# ======================================================
class EventRow(BaseModel):
    step_code: str
    start_time: str
    end_time: str


class ProcessAnalyzeCaseRequest(BaseModel):
    process_code: str
    case_id: Optional[str] = None
    events: List[EventRow]


def _get_process_artifacts(process_code: str):
    if process_code in _process_artifacts_cache:
        return _process_artifacts_cache[process_code]
    art = load_process_artifacts(
        process_code=process_code,
        model_root_dir=PROCESS_MODEL_ROOT,
        registry_dir=REGISTRY_DIR,
    )
    _process_artifacts_cache[process_code] = art
    return art


def _events_to_df(process_code: str, case_id: str, events: List[EventRow]) -> pd.DataFrame:
    return pd.DataFrame([{
        "process_code": process_code,
        "case_id": case_id,
        "step_code": e.step_code,
        "start_time": e.start_time,
        "end_time": e.end_time,
    } for e in events])


def _compute_top_step_p95_and_z(row: pd.Series, art) -> Dict[str, float]:
    """
    Pick top step by P95 ratio (duration/p95). Then compute true z-score for that SAME step.
    Returns:
      best_step_idx (1..N), best_dev (duration/p95), best_dur, best_p95, best_z
    """
    best_step_idx, best_dev, best_dur = 0, 0.0, 0.0
    best_mean, best_std, best_p95 = 0.0, 0.0, 0.0

    for i, s in enumerate(art.step_codes, start=1):
        dur = float(row.get(f"{s}_duration_min", 0.0))

        b = art.baselines.get("steps", {}).get(s, {})
        mean = float(b.get("mean", 0.0))
        std = float(b.get("std", 0.0))
        p95 = float(b.get("p95", 0.0))

        dev = (dur / p95) if p95 > 0 else 0.0

        if dev > best_dev:
            best_dev, best_step_idx, best_dur = dev, i, dur
            best_mean, best_std, best_p95 = mean, std, p95

    best_z = ((best_dur - best_mean) / best_std) if best_std > 1e-9 else 0.0

    return {
        "best_step_idx": int(best_step_idx),
        "best_dev": float(best_dev),
        "best_dur": float(best_dur),
        "best_p95": float(best_p95),
        "best_z": float(best_z),
    }


def _risk_from_quantiles(raw_anomaly: float, quantiles: List[float]) -> int:
    q = np.array(quantiles, dtype=float)
    idx = int(np.searchsorted(q, raw_anomaly, side="right") - 1)
    return int(max(0, min(100, idx)))


@app.post("/process/analyze_case_numeric")
def process_analyze_case_numeric(req: ProcessAnalyzeCaseRequest):
    """
    JSON input: events
    Output: process risk + top bottleneck (P95 ratio) + z-score (true) + durations.
    """
    if req.process_code not in PROCESS_ID_MAP:
        raise HTTPException(status_code=400, detail="Unknown process_code")

    try:
        art = _get_process_artifacts(req.process_code)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Process artifacts load failed: {e}")

    case_id = req.case_id or "CASE_UNKNOWN"
    df_case = _events_to_df(req.process_code, case_id, req.events)

    df_valid, vrep = validate_events_df(
        df_case,
        process_code=req.process_code,
        valid_steps=art.step_codes,
        allow_unknown_steps=False,
    )
    if not vrep.ok:
        raise HTTPException(status_code=400, detail={"errors": vrep.errors, "warnings": vrep.warnings})

    feat_df, _, _ = build_case_feature_matrix(
        df_valid,
        step_codes=art.step_codes,
        cases_context_df=None,
        include_context_numeric=False,
    )
    if feat_df.empty:
        raise HTTPException(status_code=400, detail="No valid features produced for case")

    row = feat_df.iloc[0]

    X = row.values.reshape(1, -1).astype(float)
    Xs = art.scaler.transform(X)

    raw_anomaly = float(-art.model.score_samples(Xs)[0])
    risk_score = _risk_from_quantiles(raw_anomaly, art.score_quantiles)
    is_anomaly = (risk_score >= 80)

    top = _compute_top_step_p95_and_z(row, art)
    best_step_idx = top["best_step_idx"]
    top_step_name = art.step_codes[best_step_idx - 1] if best_step_idx > 0 else ""

    total_process_time_min = float(row.get("total_process_time_min", 0.0))

    return {
        "process_id": PROCESS_ID_MAP[req.process_code],
        "risk_score": int(risk_score),
        "anomaly_score": round(raw_anomaly, 6),
        "is_anomaly": bool(is_anomaly),

        "top_step_name": top_step_name,

        # MAIN
        "top_step_deviation_p95": round(float(top["best_dev"]), 3),
        "top_step_p95_min": round(float(top["best_p95"]), 3),

        # SECONDARY (true z-score)
        "top_step_zscore": round(float(top["best_z"]), 3),

        "top_step_duration_min": round(float(top["best_dur"]), 3),
        "total_process_time_min": round(float(total_process_time_min), 3),
    }


@app.post("/process/analyze_case_file_numeric")
async def process_analyze_case_file_numeric(
    process_code: str,
    file: UploadFile = File(...),
    case_id: Optional[str] = None,
):
    """
    Upload CSV (event-based). Must contain 1 case_id (or provide case_id).
    Output: process risk + top bottleneck (P95 ratio) + z-score (true) + durations.
    """
    if process_code not in PROCESS_ID_MAP:
        raise HTTPException(status_code=400, detail="Unknown process_code")

    try:
        art = _get_process_artifacts(process_code)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Process artifacts load failed: {e}")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")

    df_valid, vrep = validate_events_df(
        df,
        process_code=process_code,
        valid_steps=art.step_codes,
        allow_unknown_steps=False,
    )
    if not vrep.ok:
        raise HTTPException(status_code=400, detail={"errors": vrep.errors, "warnings": vrep.warnings})

    unique_cases = df_valid["case_id"].astype(str).unique().tolist()
    if case_id is None:
        if len(unique_cases) != 1:
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain exactly 1 case_id, found {len(unique_cases)}. Provide case_id to select.",
            )
        case_id = unique_cases[0]
    else:
        case_id = str(case_id)
        if case_id not in unique_cases:
            raise HTTPException(status_code=400, detail=f"case_id not found in file: {case_id}")

    one = df_valid[df_valid["case_id"].astype(str) == case_id].copy()

    feat_df, _, _ = build_case_feature_matrix(
        one,
        step_codes=art.step_codes,
        cases_context_df=None,
        include_context_numeric=False,
    )
    if feat_df.empty:
        raise HTTPException(status_code=400, detail="No valid features produced for case")

    row = feat_df.iloc[0]

    X = row.values.reshape(1, -1).astype(float)
    Xs = art.scaler.transform(X)

    raw_anomaly = float(-art.model.score_samples(Xs)[0])
    risk_score = _risk_from_quantiles(raw_anomaly, art.score_quantiles)
    is_anomaly = (risk_score >= 80)

    top = _compute_top_step_p95_and_z(row, art)
    best_step_idx = top["best_step_idx"]
    top_step_name = art.step_codes[best_step_idx - 1] if best_step_idx > 0 else ""

    total_process_time_min = float(row.get("total_process_time_min", 0.0))

    return {
        "process_id": PROCESS_ID_MAP[process_code],
        "risk_score": int(risk_score),
        "anomaly_score": round(raw_anomaly, 6),
        "is_anomaly": bool(is_anomaly),

        "top_step_name": top_step_name,

        # MAIN
        "top_step_deviation_p95": round(float(top["best_dev"]), 3),
        "top_step_p95_min": round(float(top["best_p95"]), 3),

        # SECONDARY (true z-score)
        "top_step_zscore": round(float(top["best_z"]), 3),

        "top_step_duration_min": round(float(top["best_dur"]), 3),
        "total_process_time_min": round(float(total_process_time_min), 3),
    }