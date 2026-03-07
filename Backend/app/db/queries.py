"""
All raw SQL queries — PostgreSQL is the analytical engine.
Python handles only orchestration.
"""

# ── Ingestion (executemany-compatible) ────────────────────────────────────────

INSERT_PROCESS = """
INSERT INTO processes (process_id)
VALUES (:process_id)
ON CONFLICT (process_id) DO NOTHING;
"""

INSERT_STEP_EXECUTION = """
INSERT INTO step_executions
    (process_id, step_code, location, start_time, end_time, duration_minutes)
VALUES
    (:process_id, :step_code, :location, :start_time::timestamp, :end_time::timestamp, :duration_minutes);
"""

# ── Baseline Computation ──────────────────────────────────────────────────────

RECOMPUTE_BASELINES = """
INSERT INTO baselines (step_code, location, mean, std, p95, sample_size, updated_at)
SELECT
    step_code,
    location,
    AVG(duration_minutes)                                                   AS mean,
    COALESCE(STDDEV_POP(duration_minutes), 0)                               AS std,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_minutes)          AS p95,
    COUNT(*)                                                                AS sample_size
FROM step_executions
GROUP BY step_code, location
ON CONFLICT (step_code, location) DO UPDATE SET
    mean        = EXCLUDED.mean,
    std         = EXCLUDED.std,
    p95         = EXCLUDED.p95,
    sample_size = EXCLUDED.sample_size,
    updated_at  = now();
"""

# ── Anomaly Detection ─────────────────────────────────────────────────────────
# Clears and reinserts anomalies on each run to keep results fresh.

CLEAR_ANOMALIES = "DELETE FROM anomalies;"

DETECT_ANOMALIES = """
INSERT INTO anomalies
    (process_id, step_code, location, duration_minutes, z_score, risk_percent)
SELECT
    s.process_id,
    s.step_code,
    s.location,
    s.duration_minutes,
    (s.duration_minutes - b.mean) / NULLIF(b.std, 0)           AS z_score,
    LEAST(100.0, (s.duration_minutes / b.p95) * 100.0)         AS risk_percent
FROM step_executions s
JOIN baselines b
    ON s.step_code = b.step_code
   AND s.location  = b.location
WHERE
    s.duration_minutes > b.p95
    OR (b.std > 0 AND (s.duration_minutes - b.mean) / b.std >= 2.0);
"""

# ── Query: Anomalies List ─────────────────────────────────────────────────────

GET_ANOMALIES = """
SELECT
    id, process_id, step_code, location,
    duration_minutes, z_score, risk_percent, detected_at
FROM anomalies
WHERE (:min_risk IS NULL OR risk_percent >= :min_risk)
ORDER BY risk_percent DESC, detected_at DESC
LIMIT :limit;
"""

COUNT_ANOMALIES = """
SELECT COUNT(*) FROM anomalies
WHERE (:min_risk IS NULL OR risk_percent >= :min_risk);
"""

# ── Query: Process Detail ─────────────────────────────────────────────────────

GET_PROCESS_STEPS = """
SELECT
    s.step_code,
    s.location,
    s.start_time,
    s.end_time,
    s.duration_minutes,
    a.z_score,
    a.risk_percent,
    b.mean   AS baseline_mean,
    b.p95    AS baseline_p95
FROM step_executions s
LEFT JOIN anomalies a
       ON a.process_id       = s.process_id
      AND a.step_code        = s.step_code
      AND a.location         = s.location
      AND a.duration_minutes = s.duration_minutes
LEFT JOIN baselines b
       ON b.step_code = s.step_code
      AND b.location  = s.location
WHERE s.process_id = :process_id
ORDER BY s.start_time;
"""

# ── Query: Baselines List ─────────────────────────────────────────────────────

GET_BASELINES = """
SELECT step_code, location, mean, std, p95, sample_size, updated_at
FROM baselines
WHERE (:step_code IS NULL OR step_code = :step_code)
  AND (:location  IS NULL OR location  = :location)
ORDER BY step_code, location;
"""
