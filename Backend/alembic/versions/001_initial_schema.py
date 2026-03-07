"""Initial schema

Revision ID: 001
Revises:
Create Date: 2026-03-07
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "processes",
        sa.Column("process_id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("now()")),
    )

    op.create_table(
        "step_executions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("process_id", sa.String(), sa.ForeignKey("processes.process_id"), nullable=False),
        sa.Column("step_code", sa.String(), nullable=False),
        sa.Column("location", sa.String(), nullable=False),
        sa.Column("start_time", sa.TIMESTAMP(), nullable=False),
        sa.Column("end_time", sa.TIMESTAMP(), nullable=False),
        sa.Column("duration_minutes", sa.Double(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.text("now()")),
        sa.CheckConstraint("duration_minutes > 0", name="ck_duration_positive"),
        sa.CheckConstraint("end_time > start_time", name="ck_end_after_start"),
    )
    op.create_index("idx_step_exec_step_loc", "step_executions", ["step_code", "location"])
    op.create_index("idx_step_exec_process", "step_executions", ["process_id"])

    op.create_table(
        "baselines",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("step_code", sa.String(), nullable=False),
        sa.Column("location", sa.String(), nullable=False),
        sa.Column("mean", sa.Double(), nullable=False),
        sa.Column("std", sa.Double(), nullable=False),
        sa.Column("p95", sa.Double(), nullable=False),
        sa.Column("sample_size", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(), server_default=sa.text("now()")),
        sa.UniqueConstraint("step_code", "location", name="uq_baseline_step_loc"),
    )

    op.create_table(
        "anomalies",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("process_id", sa.String(), nullable=False),
        sa.Column("step_code", sa.String(), nullable=False),
        sa.Column("location", sa.String(), nullable=False),
        sa.Column("duration_minutes", sa.Double(), nullable=False),
        sa.Column("z_score", sa.Double(), nullable=False),
        sa.Column("risk_percent", sa.Double(), nullable=False),
        sa.Column("detected_at", sa.TIMESTAMP(), server_default=sa.text("now()")),
    )


def downgrade() -> None:
    op.drop_table("anomalies")
    op.drop_table("baselines")
    op.drop_index("idx_step_exec_process")
    op.drop_index("idx_step_exec_step_loc")
    op.drop_table("step_executions")
    op.drop_table("processes")
