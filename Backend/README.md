# Logistics Process Bottleneck Detection — Backend

FastAPI + PostgreSQL backend for detecting abnormal step durations in logistics operations using statistical baselines.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — DATABASE_URL must use psycopg (binary wheels, works on Python 3.13):
# DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/logistics_db
```
If you still have `postgresql+asyncpg://` in `.env`, change it to `postgresql+psycopg://`.

### 3. Run database migrations

```bash
alembic upgrade head
```

### 4. Start the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload CSV, trigger baseline recompute + anomaly detection |
| `GET` | `/api/anomalies` | List detected bottlenecks (`limit`, `min_risk`) |
| `GET` | `/api/process/{process_id}` | Full step timeline for one process |
| `GET` | `/api/baselines` | View statistical baselines (`step_code`, `location`) |
| `POST` | `/api/schema/suggest` | Auto-detect CSV column mapping from a file |
| `POST` | `/api/schema/suggest/columns` | Auto-detect mapping from a list of column names |
| `GET` | `/health` | Health check |

---

## Required CSV Schema

| Column | Type | Description |
|--------|------|-------------|
| `process_id` | string | Unique shipment/order ID |
| `step_code` | string | Step identifier (e.g. PICK, PACK) |
| `location` | string | Facility or warehouse code |
| `start_time` | datetime | Step start (ISO 8601) |
| `end_time` | datetime | Step end (ISO 8601) |

The schema mapper accepts flexible column names and maps them automatically.

---

## Detection Logic

A step is flagged as an anomaly when **either** is true:

- `duration_minutes > baseline p95`
- `z-score = (duration - mean) / std ≥ 2.0`

Risk categories:

| Risk % | Level |
|--------|-------|
| < 80% | Normal |
| 80–100% | Warning |
| ≥ 100% | High Risk |
