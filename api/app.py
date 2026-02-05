from fastapi import FastAPI, HTTPException, UploadFile, File
from contextlib import asynccontextmanager
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import io
from datetime import datetime
import numpy as np

# --- 1. CẤU HÌNH ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')

models = {}


# --- 2. KHỞI ĐỘNG SERVER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"📂 Loading models from: {MODEL_DIR}")
    try:
        models["driver"] = joblib.load(os.path.join(MODEL_DIR, 'driver_ai.pkl'))
        models["fleet"] = joblib.load(os.path.join(MODEL_DIR, 'fleet_ai.pkl'))
        models["ops"] = joblib.load(os.path.join(MODEL_DIR, 'ops_ai.pkl'))
        print("✅ All AI Models Loaded Successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load models. Error: {e}")
    yield


app = FastAPI(title="Logistics AI Core", lifespan=lifespan)


# --- 3. INPUT SCHEMA ---
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


def get_risk_probability(model, df_input):
    try:
        probs = model.predict_proba(df_input)[0]
        if len(probs) > 1: return probs[1]
        return probs[0] if model.classes_[0] == 1 else 0.0
    except:
        return 0.0


# --- 4. HÀM XỬ LÝ CORE (Dùng chung cho cả JSON và CSV) ---
def process_single_shipment(item: ShipmentInput):
    # --- A. CHẠY AI ---
    driver_df = pd.DataFrame([{
        'years_experience': item.years_experience,
        'total_accidents': item.total_accidents,
        'avg_ontime_rate': item.avg_ontime_rate,
        'avg_miles_per_month': item.avg_miles_per_month,
        'avg_mpg': item.avg_mpg_driver
    }])
    driver_prob = get_risk_probability(models["driver"], driver_df)

    fleet_df = pd.DataFrame([{
        'truck_age': item.truck_age,
        'lifetime_maint_cost': item.lifetime_maint_cost,
        'maint_frequency': item.maint_frequency,
        'total_downtime': item.total_downtime,
        'avg_monthly_miles': item.avg_monthly_miles_truck
    }])
    fleet_prob = get_risk_probability(models["fleet"], fleet_df)

    ops_df = pd.DataFrame([{
        'detention_hours': item.detention_hours,
        'real_mpg': item.real_mpg_trip,
        'delay_hours': item.delay_hours,
        'actual_distance_miles': item.actual_distance_miles
    }])
    ops_prob = get_risk_probability(models["ops"], ops_df)

    # --- B. GIẢ LẬP CÁC BƯỚC ---
    contributors = []
    total_duration = 0

    # STEP 1: LOADING
    step1_actual = 60 + (item.detention_hours * 60)
    step1_p95 = 120.0
    if step1_actual > step1_p95:
        contributors.append({
            "step_code": "STEP_01_LOADING",
            "actual_duration_min": round(step1_actual, 1),
            "historical_mean_min": 60.0,
            "historical_p95_min": step1_p95,
            "deviation_factor": round(step1_actual / step1_p95, 2),
            "explanation": f"Loading time ({step1_actual}m) exceeded P95 due to high detention."
        })
    total_duration += step1_actual

    # STEP 2: TRANSIT
    est_transit_time = (item.actual_distance_miles / 50) * 60
    delay_factor = 1.0 + (fleet_prob * 0.5) + (driver_prob * 0.3)
    step2_actual = est_transit_time * delay_factor
    step2_p95 = est_transit_time * 1.2
    if step2_actual > step2_p95:
        contributors.append({
            "step_code": "STEP_02_TRANSIT",
            "actual_duration_min": round(step2_actual, 1),
            "historical_mean_min": round(est_transit_time, 1),
            "historical_p95_min": round(step2_p95, 1),
            "deviation_factor": round(step2_actual / step2_p95, 2),
            "explanation": f"Transit delayed due to risk factors."
        })
    total_duration += step2_actual

    # STEP 3: UNLOADING
    step3_actual = 45 + (item.delay_hours * 30)
    step3_p95 = 90.0
    if step3_actual > step3_p95:
        contributors.append({
            "step_code": "STEP_03_UNLOADING",
            "actual_duration_min": round(step3_actual, 1),
            "historical_mean_min": 45.0,
            "historical_p95_min": step3_p95,
            "deviation_factor": round(step3_actual / step3_p95, 2),
            "explanation": f"Unloading delayed by {item.delay_hours}h."
        })
    total_duration += step3_actual

    # --- C. TỔNG HỢP ---
    contributors.sort(key=lambda x: x['deviation_factor'], reverse=True)
    for idx, c in enumerate(contributors):
        c['contribution_rank'] = idx + 1

    total_risk_score = max(driver_prob, fleet_prob, ops_prob) * 100
    is_anomaly = len(contributors) > 0

    action = "Standard Processing"
    if is_anomaly:
        top_step = contributors[0]['step_code']
        action = f"Review operational logs for {top_step}."

    return {
        "meta": {
            "process_code": "LOGISTICS_DELIVERY_FLOW",
            "case_id": str(item.case_id),
            "timestamp": datetime.now().isoformat() + "Z"
        },
        "analysis": {
            "risk_score": round(total_risk_score, 1),
            "is_anomaly": is_anomaly,
            "bottleneck_likelihood": "High" if is_anomaly else "Low"
        },
        "explainability": {
            "primary_contributors": contributors,
            "process_stats": {
                "total_duration_min": round(total_duration, 1),
                "step_count": 3
            }
        },
        "recommendation": {
            "action": action,
            "type": "Operational Deviation" if is_anomaly else "Routine Check"
        }
    }


# --- 5. ENDPOINT 1: XỬ LÝ ĐƠN LẺ (JSON) ---
@app.post("/analyze_shipment")
def analyze_shipment(item: ShipmentInput):
    if not models: raise HTTPException(status_code=500, detail="Models not loaded")
    return process_single_shipment(item)


# --- 6. ENDPOINT 2: XỬ LÝ FILE CSV -> BÁO CÁO QUY TRÌNH TỔNG HỢP ---
@app.post("/analyze_batch_csv")
async def analyze_batch_csv(file: UploadFile = File(...)):
    if not models: raise HTTPException(status_code=500, detail="Models not loaded")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")

    # Biến để thống kê
    total_cases = len(df)
    total_risk_accumulated = 0
    anomaly_count = 0

    # Thống kê lỗi theo từng bước (để tìm điểm nghẽn hệ thống)
    step_failures = {
        "STEP_01_LOADING": 0,
        "STEP_02_TRANSIT": 0,
        "STEP_03_UNLOADING": 0
    }

    # Danh sách chi tiết (nếu cần xem lại)
    detailed_results = []

    # 1. QUÉT TOÀN BỘ DỮ LIỆU
    for index, row in df.iterrows():
        try:
            # Map dữ liệu
            item = ShipmentInput(
                case_id=str(row.get('trip_id', f'ROW_{index}')),
                years_experience=row['years_experience'],
                total_accidents=row['total_accidents'],
                avg_ontime_rate=row['avg_ontime_rate'],
                avg_miles_per_month=row['avg_miles_per_month'],
                avg_mpg_driver=row['avg_mpg_driver'],
                truck_age=row['truck_age'],
                lifetime_maint_cost=row['lifetime_maint_cost'],
                maint_frequency=row['maint_frequency'],
                total_downtime=row['total_downtime'],
                avg_monthly_miles_truck=row['avg_monthly_miles_truck'],
                detention_hours=row['detention_hours'],
                real_mpg_trip=row['real_mpg_trip'],
                delay_hours=row['delay_hours'],
                actual_distance_miles=row['actual_distance_miles']
            )

            # Chạy AI cho từng dòng
            res = process_single_shipment(item)

            # Cộng dồn chỉ số
            risk = res['analysis']['risk_score']
            total_risk_accumulated += risk

            if res['analysis']['is_anomaly']:
                anomaly_count += 1
                # Xem lỗi nằm ở bước nào nhiều nhất
                contributors = res['explainability']['primary_contributors']
                if contributors:
                    top_cause = contributors[0]['step_code'] # Lấy nguyên nhân lớn nhất
                    if top_cause in step_failures:
                        step_failures[top_cause] += 1

            # Lưu gọn lại để tham chiếu
            detailed_results.append({
                "case_id": item.case_id,
                "risk": risk,
                "status": "Anomaly" if res['analysis']['is_anomaly'] else "Normal"
            })

        except Exception as e:
            continue # Bỏ qua dòng lỗi

    # 2. TÍNH TOÁN CHỈ SỐ QUY TRÌNH (PROCESS METRICS)
    avg_process_risk = round(total_risk_accumulated / total_cases, 1) if total_cases > 0 else 0
    anomaly_rate = round((anomaly_count / total_cases) * 100, 1) if total_cases > 0 else 0

    # Tìm điểm nghẽn lớn nhất (Systemic Bottleneck)
    most_failed_step = max(step_failures, key=step_failures.get)
    most_failed_count = step_failures[most_failed_step]

    # Đánh giá sức khỏe quy trình (Process Health)
    health_status = "HEALTHY"
    action_plan = "Maintain current operations."

    if avg_process_risk > 70 or anomaly_rate > 50:
        health_status = "CRITICAL"
        action_plan = f"SYSTEMIC FAILURE DETECTED. Major bottleneck at {most_failed_step}. Immediate process re-engineering required."
    elif avg_process_risk > 40 or anomaly_rate > 20:
        health_status = "AT_RISK"
        action_plan = f"Performance degrading. Focus on improving {most_failed_step}."

    # 3. TRẢ VỀ JSON TỔNG HỢP (MACRO VIEW)
    return {
        "process_audit_summary": {
            "audit_timestamp": datetime.now().isoformat(),
            "total_shipments_analyzed": total_cases,
            "process_health_grade": health_status, # HEALTHY / AT_RISK / CRITICAL
            "average_process_risk_score": avg_process_risk,
            "anomaly_rate_percent": anomaly_rate
        },
        "systemic_bottlenecks": {
            "primary_bottleneck": most_failed_step,
            "failure_count": most_failed_count,
            "impact_ratio": f"{round((most_failed_count/total_cases)*100, 1)}% of all shipments failed here."
        },
        "step_performance_breakdown": step_failures,
        "executive_recommendation": action_plan,
        # "raw_data": detailed_results # Bỏ comment nếu muốn xem chi tiết
    }