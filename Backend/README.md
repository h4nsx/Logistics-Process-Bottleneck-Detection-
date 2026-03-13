# Logistics Process Bottleneck Detection — Backend

FastAPI + PostgreSQL backend for detecting abnormal step durations in logistics operations using both statistical baselines and ML-based anomaly detection (IsolationForest).

---

## Stack

| Component | Technology |
|-----------|-----------|
| Web framework | FastAPI |
| Database | PostgreSQL (asyncpg driver) |
| ORM / Query | SQLAlchemy Core (async) |
| Migrations | Alembic |
| ML Inference | scikit-learn IsolationForest + joblib |

---

## Quick Start (Local — Windows)

### 1. Install dependencies

```bash
cd Backend
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set your PostgreSQL password
```

`.env` minimum:
```
DATABASE_URL=postgresql+asyncpg://postgres:YOUR_PASSWORD@localhost:5432/logistics_db
```

### 3. Run database migrations

```bash
alembic upgrade head
```

### 4. Start the server

**Windows** (use `run.py` to avoid ProactorEventLoop issue):
```bash
python run.py
```

**Linux / macOS / Docker:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs

---

## API Endpoints

### Statistical (baseline-based)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload CSV → recompute baselines + detect anomalies |
| `GET` | `/api/anomalies` | List detected bottlenecks (`limit`, `min_risk`) |
| `GET` | `/api/baselines` | View statistical baselines (`step_code`, `location`) |
| `GET` | `/api/process/{process_id}` | Full step timeline for one process |
| `POST` | `/api/schema/suggest` | Auto-detect CSV column mapping from a file |
| `POST` | `/api/schema/suggest/columns` | Auto-detect mapping from a list of column names |

### ML (IsolationForest)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/ml/analyze` | Upload CSV → ML anomaly detection + store predictions |
| `GET` | `/api/ml/predictions` | Query stored ML predictions (`process_code`, `only_anomalies`) |
| `GET` | `/api/ml/status` | Check loaded ML models |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |

---

## Required CSV Schema

| Column | Type | Description |
|--------|------|-------------|
| `process_id` | string | Unique shipment/order ID |
| `step_code` | string | Step identifier (e.g. STEP_001_PICKUP) |
| `location` | string | Facility or warehouse code |
| `start_time` | datetime | Step start (ISO 8601) |
| `end_time` | datetime | Step end (ISO 8601) |

The schema mapper accepts flexible column names and maps them automatically.

---

## Detection Logic

### Statistical detection (per step)
A step is flagged as anomaly when **either** is true:
- `duration_minutes > baseline p95`
- `z-score = (duration − mean) / std ≥ 2.0`

### ML detection (per case/shipment)
- IsolationForest trained per process type (`TRUCKING_DELIVERY_FLOW`, `IMPORT_CUSTOMS_CLEARANCE`, `WAREHOUSE_FULFILLMENT`)
- Features: step durations + aggregate stats (total time, step count, etc.)
- Output: `risk_percentile` (0–100%) + `bottleneck_steps`

---

## Database Tables

| Table | Description |
|-------|-------------|
| `processes` | Registered process IDs |
| `step_executions` | Raw event data (process + step + duration) |
| `baselines` | mean / std / p95 per step+location |
| `anomalies` | Statistical anomaly detections |
| `ml_predictions` | ML anomaly detection results |

---

## Deploy

See [DEPLOY_RENDER.md](DEPLOY_RENDER.md) for Render deployment guide.
