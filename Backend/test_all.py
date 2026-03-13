"""
End-to-end integration test for the Logistics Bottleneck Detection API.

Run from Backend/ directory:
    python test_all.py

Requires:
    - Server running on localhost:8000  (uvicorn app.main:app --reload)
    - PostgreSQL connected
    - pip install requests
"""
import io
import json
import sys
import textwrap

# Force UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import requests

BASE = "http://localhost:8000"
PASS = "[PASS]"
FAIL = "[FAIL]"

results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    icon = PASS if condition else FAIL
    print(f"  {icon} {name}" + (f"  -> {detail}" if detail else ""))
    results.append((name, condition, detail))
    return condition


def section(title: str) -> None:
    print(f"\n{'-'*55}")
    print(f"  {title}")
    print(f"{'-'*55}")


# ── 1. Health ──────────────────────────────────────────────────────────────────
section("1. Health Check")
try:
    r = requests.get(f"{BASE}/health", timeout=5)
    check("GET /health -> 200", r.status_code == 200)
    check("version field present", "version" in r.json())
except Exception as e:
    check("GET /health reachable", False, str(e))
    print("\n  [!] Server not running. Start it with:")
    print("     .\\venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000")
    sys.exit(1)


# ── 2. ML Status ───────────────────────────────────────────────────────────────
section("2. ML Model Status")
r = requests.get(f"{BASE}/api/ml/status")
check("GET /api/ml/status → 200", r.status_code == 200)
data = r.json()
loaded = data.get("loaded_models", [])
check("TRUCKING_DELIVERY_FLOW loaded",  "TRUCKING_DELIVERY_FLOW"  in loaded)
check("IMPORT_CUSTOMS_CLEARANCE loaded","IMPORT_CUSTOMS_CLEARANCE" in loaded)
check("WAREHOUSE_FULFILLMENT loaded",   "WAREHOUSE_FULFILLMENT"   in loaded)
print(f"  Model dir: {data.get('model_base_dir','')}")


# ── 3. Schema Suggest ─────────────────────────────────────────────────────────
section("3. Schema Suggestion Engine")

sample_columns = json.dumps({"columns": [
    "order_id", "step", "warehouse", "start", "finish"
]})
r = requests.post(
    f"{BASE}/api/schema/suggest/columns",
    data=sample_columns,
    headers={"Content-Type": "application/json"},
)
check("POST /api/schema/suggest/columns -> 200", r.status_code == 200)
data = r.json()
suggestions = data.get("suggestions", {})
missing = data.get("missing_required", [])
check("Has suggestions", len(suggestions) > 0, f"{len(suggestions)} columns mapped")
check("No required fields missing", len(missing) == 0, f"missing: {missing}" if missing else "all mapped")


# ── 4. Upload Statistical CSV ─────────────────────────────────────────────────
section("4. Statistical Upload  (POST /api/upload)")

STATISTICAL_CSV = textwrap.dedent("""\
    process_id,step_code,location,start_time,end_time
    ORD-TEST-001,PICK,WH-HCM,2024-01-10T08:00:00,2024-01-10T09:30:00
    ORD-TEST-001,PACK,WH-HCM,2024-01-10T09:30:00,2024-01-10T10:00:00
    ORD-TEST-001,SHIP,WH-HCM,2024-01-10T10:00:00,2024-01-10T11:00:00
    ORD-TEST-002,PICK,WH-HCM,2024-01-10T08:00:00,2024-01-10T11:00:00
    ORD-TEST-002,PACK,WH-HCM,2024-01-10T11:00:00,2024-01-10T11:30:00
    ORD-TEST-002,SHIP,WH-HCM,2024-01-10T11:30:00,2024-01-10T12:30:00
    ORD-TEST-003,PICK,WH-HCM,2024-01-10T08:00:00,2024-01-10T08:45:00
    ORD-TEST-003,PACK,WH-HCM,2024-01-10T08:45:00,2024-01-10T09:15:00
    ORD-TEST-003,SHIP,WH-HCM,2024-01-10T09:15:00,2024-01-10T10:00:00
    BAD-ROW,,WH-HCM,2024-01-10T10:00:00,2024-01-10T09:00:00
""")

r = requests.post(
    f"{BASE}/api/upload",
    files={"file": ("test_data.csv", io.BytesIO(STATISTICAL_CSV.encode()), "text/csv")},
)
check("POST /api/upload → 200", r.status_code == 200)
data = r.json()
check("status = success", data.get("status") == "success")
check("processed_rows = 9", data.get("processed_rows") == 9, str(data.get("processed_rows")))
check("invalid_rows ≥ 1", data.get("invalid_rows", 0) >= 1, str(data.get("invalid_rows")))
check("baselines_updated = true", data.get("baselines_updated") is True)
check("anomalies_detected ≥ 0", isinstance(data.get("anomalies_detected"), int),
      str(data.get("anomalies_detected")))
print(f"  anomalies_detected: {data.get('anomalies_detected')}  |  "
      f"time: {data.get('processing_time_seconds')}s")


# ── 5. Baselines ───────────────────────────────────────────────────────────────
section("5. Baselines  (GET /api/baselines)")
r = requests.get(f"{BASE}/api/baselines")
check("GET /api/baselines → 200", r.status_code == 200)
baselines = r.json().get("baselines", [])
check("Baselines exist after upload", len(baselines) > 0, f"{len(baselines)} rows")

r2 = requests.get(f"{BASE}/api/baselines?step_code=PICK&location=WH-HCM")
check("Filter by step_code+location works",
      r2.status_code == 200 and len(r2.json()["baselines"]) > 0)
bl = r2.json()["baselines"][0]
check("Baseline has mean/std/p95", all(k in bl for k in ["mean","std","p95","sample_size"]),
      f"mean={bl.get('mean'):.1f} p95={bl.get('p95'):.1f}")


# ── 6. Anomalies ───────────────────────────────────────────────────────────────
section("6. Anomalies  (GET /api/anomalies)")
r = requests.get(f"{BASE}/api/anomalies")
check("GET /api/anomalies → 200", r.status_code == 200)
data = r.json()
check("Response has anomalies + total_count", "anomalies" in data and "total_count" in data)
print(f"  Total anomalies: {data.get('total_count')}")

r2 = requests.get(f"{BASE}/api/anomalies?min_risk=80")
check("Filter min_risk=80 works", r2.status_code == 200)


# ── 7. Process Detail ──────────────────────────────────────────────────────────
section("7. Process Detail  (GET /api/process/{id})")
r = requests.get(f"{BASE}/api/process/ORD-TEST-001")
check("GET /api/process/ORD-TEST-001 → 200", r.status_code == 200)
data = r.json()
check("process_id matches", data.get("process_id") == "ORD-TEST-001")
check("steps list present", isinstance(data.get("steps"), list) and len(data["steps"]) > 0,
      f"{len(data.get('steps',[]))} steps")
check("total_duration_minutes > 0", data.get("total_duration_minutes", 0) > 0)
check("Each step has baseline_mean", all("baseline_mean" in s for s in data["steps"]))

r404 = requests.get(f"{BASE}/api/process/NONEXISTENT-999")
check("GET /api/process/NONEXISTENT → 404", r404.status_code == 404)


# ── 8. ML Analyze ─────────────────────────────────────────────────────────────
section("8. ML Analyze  (POST /api/ml/analyze)")

# Sample with 2 normal cases + 1 anomalous (very long steps)
ML_CSV = textwrap.dedent("""\
    process_code,case_id,step_code,start_time,end_time
    TRUCKING_DELIVERY_FLOW,TRK-NORMAL-001,STEP_001_DISPATCH_CREATED,2025-01-01T08:00:00,2025-01-01T08:12:00
    TRUCKING_DELIVERY_FLOW,TRK-NORMAL-001,STEP_002_CARRIER_TENDERED,2025-01-01T08:12:00,2025-01-01T08:24:00
    TRUCKING_DELIVERY_FLOW,TRK-NORMAL-001,STEP_013_EN_ROUTE_HUB,2025-01-01T09:00:00,2025-01-01T11:30:00
    TRUCKING_DELIVERY_FLOW,TRK-ANOMALY-002,STEP_001_DISPATCH_CREATED,2025-01-01T08:00:00,2025-01-01T08:12:00
    TRUCKING_DELIVERY_FLOW,TRK-ANOMALY-002,STEP_013_EN_ROUTE_HUB,2025-01-01T09:00:00,2025-01-02T05:00:00
    TRUCKING_DELIVERY_FLOW,TRK-ANOMALY-002,STEP_018_LINEHAUL_START,2025-01-02T05:00:00,2025-01-03T01:00:00
    WAREHOUSE_FULFILLMENT,WH-NORMAL-001,STEP_001_ORDER_CONFIRMED,2025-01-01T08:00:00,2025-01-01T08:10:00
    WAREHOUSE_FULFILLMENT,WH-NORMAL-001,STEP_002_FRAUD_CHECK,2025-01-01T08:10:00,2025-01-01T08:22:00
    WAREHOUSE_FULFILLMENT,WH-NORMAL-001,STEP_006_PICK_START,2025-01-01T08:22:00,2025-01-01T10:00:00
    UNKNOWN_PROCESS,SKIP-001,STEP_001,2025-01-01T08:00:00,2025-01-01T09:00:00
""")

r = requests.post(
    f"{BASE}/api/ml/analyze",
    files={"file": ("events_test.csv", io.BytesIO(ML_CSV.encode()), "text/csv")},
)
check("POST /api/ml/analyze → 200", r.status_code == 200)
data = r.json()
check("status = success", data.get("status") == "success")
check("total_cases = 3 (UNKNOWN_PROCESS skipped)",
      data.get("total_cases") == 3, str(data.get("total_cases")))
check("skipped_cases = 1", data.get("skipped_cases") == 1, str(data.get("skipped_cases")))
check("anomaly_count ≥ 1", data.get("anomaly_count", 0) >= 1,
      str(data.get("anomaly_count")))
check("predictions list present", len(data.get("predictions", [])) == 3)

preds = {p["case_id"]: p for p in data.get("predictions", [])}
if "TRK-ANOMALY-002" in preds:
    anomaly_pred = preds["TRK-ANOMALY-002"]
    check("TRK-ANOMALY-002 flagged as anomaly",
          anomaly_pred.get("is_anomaly") is True,
          f"risk={anomaly_pred.get('risk_percentile')}%")
    check("Bottleneck steps identified",
          len(anomaly_pred.get("bottleneck_steps", [])) > 0,
          str([s["step_code"] for s in anomaly_pred.get("bottleneck_steps", [])]))

print(f"  Process breakdown: {data.get('process_breakdown')}")
print(f"  Time: {data.get('processing_time_seconds')}s")


# ── 9. ML Predictions Query ────────────────────────────────────────────────────
section("9. ML Predictions Query  (GET /api/ml/predictions)")
r = requests.get(f"{BASE}/api/ml/predictions")
check("GET /api/ml/predictions → 200", r.status_code == 200)
data = r.json()
check("Has predictions + total_count", "predictions" in data and "total_count" in data)
check("Predictions stored from ml/analyze", data.get("total_count", 0) >= 3,
      f"{data.get('total_count')} records")

r2 = requests.get(f"{BASE}/api/ml/predictions?only_anomalies=true&process_code=TRUCKING_DELIVERY_FLOW")
check("Filter anomalies + process_code works", r2.status_code == 200)
filtered = r2.json()["predictions"]
if filtered:
    check("Filtered results are anomalies",
          all(p["is_anomaly"] for p in filtered))
    check("Filtered results are TRUCKING_DELIVERY_FLOW",
          all(p["process_code"] == "TRUCKING_DELIVERY_FLOW" for p in filtered))


# ── Summary ────────────────────────────────────────────────────────────────────
section("SUMMARY")
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
failed_tests = [(name, detail) for name, ok, detail in results if not ok]

print(f"\n  Result: {passed}/{total} checks passed\n")

if failed_tests:
    print("  Failed:")
    for name, detail in failed_tests:
        print(f"    {FAIL} {name}" + (f"  -> {detail}" if detail else ""))
else:
    print(f"  {PASS} All checks passed! Backend + Database + ML are working correctly.")

print()
