"""Record when the last full scan completed.

Revision ID: 20261114_0004
Revises: 20261113_0003
Create Date: 2026-11-14

排程原本用 `source_items.last_seen_at` 的最大值當最後一次掃描時間，但單檔 `ingest` 也會更新它，
啟動時的補掃因此可能被跳過。這張表只有一列，只有完整掃描在同一個 transaction 裡寫入。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261114_0004"
down_revision: str | None = "20261113_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "helpdesk"


def upgrade() -> None:
    op.create_table(
        "last_full_scan",
        sa.Column("singleton", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("singleton", name="pk_last_full_scan"),
        sa.CheckConstraint("singleton", name="ck_last_full_scan_singleton"),
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("last_full_scan", schema=SCHEMA)
