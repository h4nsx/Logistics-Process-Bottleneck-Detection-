from sqlalchemy import (
    TIMESTAMP,
    CheckConstraint,
    Column,
    Double,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    func,
)

metadata = MetaData()

processes = Table(
    "processes",
    metadata,
    Column("process_id", String, primary_key=True),
    Column("created_at", TIMESTAMP, server_default=func.now()),
)

step_executions = Table(
    "step_executions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("process_id", String, nullable=False),
    Column("step_code", String, nullable=False),
    Column("location", String, nullable=False),
    Column("start_time", TIMESTAMP, nullable=False),
    Column("end_time", TIMESTAMP, nullable=False),
    Column("duration_minutes", Double, nullable=False),
    Column("created_at", TIMESTAMP, server_default=func.now()),
    CheckConstraint("duration_minutes > 0", name="ck_duration_positive"),
    CheckConstraint("end_time > start_time", name="ck_end_after_start"),
    Index("idx_step_exec_step_loc", "step_code", "location"),
    Index("idx_step_exec_process", "process_id"),
)

baselines = Table(
    "baselines",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("step_code", String, nullable=False),
    Column("location", String, nullable=False),
    Column("mean", Double, nullable=False),
    Column("std", Double, nullable=False),
    Column("p95", Double, nullable=False),
    Column("sample_size", Integer, nullable=False),
    Column("updated_at", TIMESTAMP, server_default=func.now(), onupdate=func.now()),
    UniqueConstraint("step_code", "location", name="uq_baseline_step_loc"),
)

anomalies = Table(
    "anomalies",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("process_id", String, nullable=False),
    Column("step_code", String, nullable=False),
    Column("location", String, nullable=False),
    Column("duration_minutes", Double, nullable=False),
    Column("z_score", Double, nullable=False),
    Column("risk_percent", Double, nullable=False),
    Column("detected_at", TIMESTAMP, server_default=func.now()),
)

ml_predictions = Table(
    "ml_predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("case_id", String, nullable=False),
    Column("process_code", String, nullable=False),
    Column("anomaly_score", Double, nullable=False),
    Column("risk_percentile", Double, nullable=False),
    Column("is_anomaly", String, nullable=False),  # "true"/"false" for DB compatibility
    Column("bottleneck_steps", String, nullable=False, server_default="[]"),
    Column("total_duration_min", Double, nullable=False),
    Column("step_count", Integer, nullable=False),
    Column("analyzed_at", TIMESTAMP, server_default=func.now()),
    Index("idx_ml_pred_case", "case_id"),
    Index("idx_ml_pred_process", "process_code"),
)
