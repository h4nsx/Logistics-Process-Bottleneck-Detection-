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

# Legacy models (optional)
models: Dict[str, Any] = {}

# Process artifacts cache
_process_artifacts_cache: Dict[str, Any] = {}

# process_code -> process_id
PROCESS_ID_MAP = {
    "TRUCKING_DELIVERY_FLOW": 1,
    "WAREHOUSE_FULFILLMENT": 2,
    "IMPORT_CUSTOMS_CLEARANCE": 3,
}

# process_code -> batch wrapper key
PROCESS_BATCH_KEY_MAP = {
    "TRUCKING_DELIVERY_FLOW": "trucking_result",
    "WAREHOUSE_FULFILLMENT": "warehouse_result",
    "IMPORT_CUSTOMS_CLEARANCE": "customs_result",
}


# ======================================================
# Lifespan
# ======================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"📂 Loading legacy models from: {MODEL_DIR}")
    try:
        driver_path = os.path.join(MODEL_DIR, "driver_ai.pkl")
        fleet_path = os.path.join(MODEL_DIR, "fleet_ai.pkl")
        ops_path = os.path.join(MODEL_DIR, "ops_ai.pkl")

        if os.path.exists(driver_path):
            models["driver"] = joblib.load(driver_path)
        if os.path.exists(fleet_path):
            models["fleet"] = joblib.load(fleet_path)
        if os.path.exists(ops_path):
            models["ops"] = joblib.load(ops_path)

        print("✅ Legacy driver/fleet/ops models loaded (if present).")
    except Exception as e:
        print(f"⚠️ Warning: Could not load legacy models. Error: {e}")
    yield


app = FastAPI(title="Logistics AI Core", lifespan=lifespan)


# ======================================================
# Legacy driver/fleet/ops
# ======================================================
class ShipmentInput(BaseModel):
    case_id: str

    years_experience: int
    total_accidents: int
    avg_ontime_rate: float
    avg_miles_per_month: float
    avg_mpg_driver: float

    truck_age: int
    lifetime_maint_cost: float
    maint_frequency: int
    total_downtime: float
    avg_monthly_miles_truck: float

    detention_hours: float
    real_mpg_trip: float
    delay_hours: float
    actual_distance_miles: float


def get_risk_probability(model, df_input: pd.DataFrame) -> float:
    try:
        probs = model.predict_proba(df_input)[0]
        if len(probs) > 1:
            return float(probs[1])
        return float(probs[0])
    except Exception:
        return 0.0


def process_single_shipment(item: ShipmentInput) -> Dict[str, Any]:
    if not models:
        raise HTTPException(status_code=500, detail="Legacy models not loaded")

    driver_df = pd.DataFrame([{
        "years_experience": item.years_experience,
        "total_accidents": item.total_accidents,
        "avg_ontime_rate": item.avg_ontime_rate,
        "avg_miles_per_month": item.avg_miles_per_month,
        "avg_mpg": item.avg_mpg_driver,
    }])
    driver_prob = get_risk_probability(models["driver"], driver_df)

    fleet_df = pd.DataFrame([{
        "truck_age": item.truck_age,
        "lifetime_maint_cost": item.lifetime_maint_cost,
        "maint_frequency": item.maint_frequency,
        "total_downtime": item.total_downtime,
        "avg_monthly_miles": item.avg_monthly_miles_truck,
    }])
    fleet_prob = get_risk_probability(models["fleet"], fleet_df)

    ops_df = pd.DataFrame([{
        "detention_hours": item.detention_hours,
        "real_mpg": item.real_mpg_trip,
        "delay_hours": item.delay_hours,
        "actual_distance_miles": item.actual_distance_miles,
    }])
    ops_prob = get_risk_probability(models["ops"], ops_df)

    contributors = []
    total_duration = 0.0

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
        "analysis": {
            "risk_score": round(total_risk_score, 1),
            "is_anomaly": bool(is_anomaly),
        },
        "explainability": {
            "contributors": contributors,
            "total_duration_min": round(total_duration, 1),
        },
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
# Process bottleneck models
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


def _risk_from_quantiles(raw_anomaly: float, quantiles: list) -> int:
    q = np.array(quantiles, dtype=float)
    idx = int(np.searchsorted(q, raw_anomaly, side="right") - 1)
    return int(max(0, min(100, idx)))


def _compute_top_step_p95_and_z(row: pd.Series, art) -> Dict[str, float]:
    """
    Pick top step by P95 ratio, then compute z-score for that same step.
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


def _extract_step_index(step_name: str) -> int:
    try:
        parts = str(step_name).split("_")
        if len(parts) >= 2:
            return int(parts[1])
    except Exception:
        pass
    return 0


def compute_process_specific(process_code: str, case_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Add 3 process-specific metrics for each process (single case).
    """
    case_df = case_df.copy()
    case_df["start_time"] = pd.to_datetime(case_df["start_time"])
    case_df["end_time"] = pd.to_datetime(case_df["end_time"])
    case_df["duration_min"] = (case_df["end_time"] - case_df["start_time"]).dt.total_seconds() / 60.0

    if process_code == "TRUCKING_DELIVERY_FLOW":
        transit_delay_min = float(
            case_df.loc[
                case_df["step_code"].astype(str).str.contains("TRANSIT|LINEHAUL|EN_ROUTE", case=False, regex=True),
                "duration_min"
            ].sum()
        )

        hub_touch_count = int(
            case_df["step_code"].astype(str).str.contains("HUB", case=False, regex=False).sum()
        )

        delivery_attempt_count = int(
            case_df["step_code"].astype(str).str.contains("DELIVERY_ATTEMPT", case=False, regex=False).sum()
        )

        return {
            "transit_delay_min": round(transit_delay_min, 3),
            "hub_touch_count": hub_touch_count,
            "delivery_attempt_count": delivery_attempt_count,
        }

    if process_code == "WAREHOUSE_FULFILLMENT":
        pick_pack_time_min = float(
            case_df.loc[
                case_df["step_code"].astype(str).str.contains("PICK|PACK", case=False, regex=True),
                "duration_min"
            ].sum()
        )

        qc_rework_flag = int(
            case_df["step_code"].astype(str).str.contains("REWORK|RECHECK", case=False, regex=True).any()
        )

        staging_wait_min = float(
            case_df.loc[
                case_df["step_code"].astype(str).str.contains("STAGING|DOCK_ASSIGN", case=False, regex=True),
                "duration_min"
            ].sum()
        )

        return {
            "pick_pack_time_min": round(pick_pack_time_min, 3),
            "qc_rework_flag": qc_rework_flag,
            "staging_wait_min": round(staging_wait_min, 3),
        }

    if process_code == "IMPORT_CUSTOMS_CLEARANCE":
        inspection_delay_min = float(
            case_df.loc[
                case_df["step_code"].astype(str).str.contains("INSPECTION", case=False, regex=False),
                "duration_min"
            ].sum()
        )

        document_recheck_flag = int(
            case_df["step_code"].astype(str).str.contains(
                "ADDITIONAL_DOCS|AMENDMENT|RECHECK|DOC_VALIDATION",
                case=False,
                regex=True
            ).any()
        )

        submit_rows = case_df[
            case_df["step_code"].astype(str).str.contains("SUBMIT_DECLARATION", case=False, regex=False)
        ]
        release_rows = case_df[
            case_df["step_code"].astype(str).str.contains("RELEASED|FINAL_CLEARANCE", case=False, regex=True)
        ]

        clearance_cycle_time_min = 0.0
        if not submit_rows.empty and not release_rows.empty:
            start_submit = submit_rows["start_time"].min()
            end_release = release_rows["end_time"].max()
            clearance_cycle_time_min = (end_release - start_submit).total_seconds() / 60.0

        return {
            "inspection_delay_min": round(inspection_delay_min, 3),
            "document_recheck_flag": document_recheck_flag,
            "clearance_cycle_time_min": round(clearance_cycle_time_min, 3),
        }

    return {}


def _build_final_output(
    process_code: str,
    raw_anomaly: float,
    risk_score: int,
    row: pd.Series,
    art,
    case_df: pd.DataFrame,
) -> Dict[str, Any]:
    is_anomaly = bool(risk_score >= 80)

    top = _compute_top_step_p95_and_z(row, art)
    best_step_idx = top["best_step_idx"]
    top_step_name = art.step_codes[best_step_idx - 1] if best_step_idx > 0 else ""

    total_process_time_min = float(row.get("total_process_time_min", 0.0))
    process_specific = compute_process_specific(process_code, case_df)

    return {
        "process_id": PROCESS_ID_MAP[process_code],
        "risk_score": int(risk_score),
        "anomaly_score": round(float(raw_anomaly), 6),
        "is_anomaly": bool(is_anomaly),
        "top_step_name": top_step_name,
        "top_step_deviation_p95": round(float(top["best_dev"]), 3),
        "top_step_p95_min": round(float(top["best_p95"]), 3),
        "top_step_zscore": round(float(top["best_z"]), 3),
        "top_step_duration_min": round(float(top["best_dur"]), 3),
        "total_process_time_min": round(float(total_process_time_min), 3),
        "process_specific": process_specific,
    }


def _analyze_single_case_df(process_code: str, case_df: pd.DataFrame, art) -> Optional[Dict[str, Any]]:
    """
    Analyze exactly one case DataFrame and return final unified output.
    """
    feat_df, _, _ = build_case_feature_matrix(
        case_df,
        step_codes=art.step_codes,
        cases_context_df=None,
        include_context_numeric=False,
    )

    if feat_df.empty:
        return None

    row = feat_df.iloc[0]
    X = row.values.reshape(1, -1).astype(float)
    Xs = art.scaler.transform(X)

    raw_anomaly = float(-art.model.score_samples(Xs)[0])
    risk_score = _risk_from_quantiles(raw_anomaly, art.score_quantiles)

    result = _build_final_output(process_code, raw_anomaly, risk_score, row, art, case_df)
    result["case_id"] = str(case_df["case_id"].astype(str).iloc[0])
    return result


def _build_batch_output(process_code: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build compact numeric-only batch output for one process.
    Wrapper key:
    - trucking_result
    - warehouse_result
    - customs_result
    """
    if not results:
        raise ValueError("results is empty")

    case_count = len(results)
    anomaly_count = int(sum(1 for r in results if r["is_anomaly"]))
    avg_risk_score = round(float(np.mean([r["risk_score"] for r in results])), 3)
    avg_anomaly_score = round(float(np.mean([r["anomaly_score"] for r in results])), 6)
    avg_total_process_time_min = round(float(np.mean([r["total_process_time_min"] for r in results])), 3)

    # dominant step from top_step_name
    step_counter: Dict[str, int] = {}
    step_duration_map: Dict[str, List[float]] = {}

    for r in results:
        step_name = r["top_step_name"]
        step_counter[step_name] = step_counter.get(step_name, 0) + 1
        step_duration_map.setdefault(step_name, []).append(float(r["top_step_duration_min"]))

    dominant_step_name = max(step_counter, key=step_counter.get)
    dominant_step_index = _extract_step_index(dominant_step_name)
    dominant_step_case_count = int(step_counter[dominant_step_name])
    dominant_step_case_rate = round(dominant_step_case_count / max(1, case_count), 4)
    avg_dominant_step_duration_min = round(float(np.mean(step_duration_map[dominant_step_name])), 3)

    # process_specific batch aggregation
    if process_code == "TRUCKING_DELIVERY_FLOW":
        process_specific = {
            "avg_transit_delay_min": round(float(np.mean([r["process_specific"]["transit_delay_min"] for r in results])), 3),
            "avg_hub_touch_count": round(float(np.mean([r["process_specific"]["hub_touch_count"] for r in results])), 3),
            "avg_delivery_attempt_count": round(float(np.mean([r["process_specific"]["delivery_attempt_count"] for r in results])), 3),
        }

    elif process_code == "WAREHOUSE_FULFILLMENT":
        process_specific = {
            "avg_pick_pack_time_min": round(float(np.mean([r["process_specific"]["pick_pack_time_min"] for r in results])), 3),
            "qc_rework_rate": round(float(np.mean([r["process_specific"]["qc_rework_flag"] for r in results])), 4),
            "avg_staging_wait_min": round(float(np.mean([r["process_specific"]["staging_wait_min"] for r in results])), 3),
        }

    elif process_code == "IMPORT_CUSTOMS_CLEARANCE":
        process_specific = {
            "avg_inspection_delay_min": round(float(np.mean([r["process_specific"]["inspection_delay_min"] for r in results])), 3),
            "document_recheck_rate": round(float(np.mean([r["process_specific"]["document_recheck_flag"] for r in results])), 4),
            "avg_clearance_cycle_time_min": round(float(np.mean([r["process_specific"]["clearance_cycle_time_min"] for r in results])), 3),
        }

    else:
        process_specific = {}

    payload = {
        "process_id": PROCESS_ID_MAP[process_code],
        "case_count": int(case_count),
        "avg_risk_score": avg_risk_score,
        "avg_anomaly_score": avg_anomaly_score,
        "anomaly_count": anomaly_count,
        "anomaly_rate": round(anomaly_count / max(1, case_count), 4),
        "dominant_step_index": int(dominant_step_index),
        "dominant_step_case_count": dominant_step_case_count,
        "dominant_step_case_rate": dominant_step_case_rate,
        "avg_dominant_step_duration_min": avg_dominant_step_duration_min,
        "avg_total_process_time_min": avg_total_process_time_min,
        "process_specific": process_specific,
    }

    wrapper_key = PROCESS_BATCH_KEY_MAP[process_code]
    return {wrapper_key: payload}


@app.post("/process/analyze_case_numeric")
def process_analyze_case_numeric(req: ProcessAnalyzeCaseRequest):
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

    return _build_final_output(req.process_code, raw_anomaly, risk_score, row, art, df_valid)


@app.post("/process/analyze_case_file_numeric")
async def process_analyze_case_file_numeric(
    process_code: str,
    file: UploadFile = File(...),
    case_id: Optional[str] = None,
):
    """
    Analyze exactly one shipment/case from uploaded CSV.
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

    return _build_final_output(process_code, raw_anomaly, risk_score, row, art, one)


@app.post("/process/analyze_batch_file_numeric")
async def process_analyze_batch_file_numeric(
    process_code: str,
    file: UploadFile = File(...),
    max_cases: Optional[int] = None,
):
    """
    Analyze many shipments/cases from uploaded CSV.
    Returns one numeric-only block:
    - trucking_result
    - warehouse_result
    - customs_result
    depending on process_code.
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

    df_valid = df_valid[df_valid["process_code"].astype(str) == process_code].copy()
    if df_valid.empty:
        raise HTTPException(status_code=400, detail=f"No rows found for process_code={process_code}")

    case_ids = df_valid["case_id"].astype(str).drop_duplicates().tolist()
    if max_cases is not None:
        case_ids = case_ids[:max_cases]

    results: List[Dict[str, Any]] = []

    for cid in case_ids:
        one = df_valid[df_valid["case_id"].astype(str) == cid].copy()
        try:
            out = _analyze_single_case_df(process_code, one, art)
            if out is not None:
                results.append(out)
        except Exception:
            continue

    if not results:
        raise HTTPException(status_code=400, detail="No valid cases analyzed from file")

    return _build_batch_output(process_code, results)