"""Add ml_predictions table

Revision ID: 002
Revises: 001
Create Date: 2026-03-12
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ml_predictions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("case_id", sa.String(), nullable=False),
        sa.Column("process_code", sa.String(), nullable=False),
        sa.Column("anomaly_score", sa.Double(), nullable=False),
        sa.Column("risk_percentile", sa.Double(), nullable=False),
        # Stored as "true"/"false" string for broad DB compatibility
        sa.Column("is_anomaly", sa.String(), nullable=False),
        sa.Column("bottleneck_steps", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("total_duration_min", sa.Double(), nullable=False),
        sa.Column("step_count", sa.Integer(), nullable=False),
        sa.Column("analyzed_at", sa.TIMESTAMP(), server_default=sa.text("now()")),
    )
    op.create_index("idx_ml_pred_case", "ml_predictions", ["case_id"])
    op.create_index("idx_ml_pred_process", "ml_predictions", ["process_code"])
    op.create_index("idx_ml_pred_risk", "ml_predictions", ["risk_percentile"])


def downgrade() -> None:
    op.drop_index("idx_ml_pred_risk")
    op.drop_index("idx_ml_pred_process")
    op.drop_index("idx_ml_pred_case")
    op.drop_table("ml_predictions")
