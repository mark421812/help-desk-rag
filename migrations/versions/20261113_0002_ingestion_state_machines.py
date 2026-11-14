"""Constrain source item and ingestion job states to their domain vocabularies.

Revision ID: 20261113_0002
Revises: 20261113_0001
Create Date: 2026-11-13

第一個 revision 把 source item 與 job 的狀態留成自由字串；這裡補上 CHECK，並加入
`change_generation`：掃描每看到一次變化就加一，處理 job 的 idempotency key 由它組成。
允許值寫成字面值，理由同第一個 revision。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261113_0002"
down_revision: str | None = "20261113_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "helpdesk"


def _one_of(column: str, values: Sequence[str]) -> str:
    return f"{column} IN ({', '.join(repr(value) for value in values)})"


_CHECKS: tuple[tuple[str, str, str], ...] = (
    (
        "source_items",
        "ck_source_items_status",
        _one_of(
            "status",
            (
                "pending",
                "processing",
                "completed",
                "metadata_missing",
                "ocr_required",
                "quarantined",
                "failed",
                "deleted",
            ),
        ),
    ),
    (
        "source_items",
        "ck_source_items_deleted_at",
        "(status = 'deleted') = (deleted_at IS NOT NULL)",
    ),
    (
        "source_items",
        "ck_source_items_error",
        "(error_code IS NULL) = (error_message IS NULL)",
    ),
    ("source_items", "ck_source_items_generation", "change_generation >= 0"),
    (
        "ingestion_jobs",
        "ck_ingestion_jobs_status",
        _one_of("status", ("pending", "processing", "completed", "failed")),
    ),
    (
        "ingestion_jobs",
        "ck_ingestion_jobs_job_type",
        _one_of("job_type", ("process_item", "delete_item")),
    ),
    (
        "ingestion_jobs",
        "ck_ingestion_jobs_lease",
        "(status = 'processing') = (leased_by IS NOT NULL) "
        "AND (leased_by IS NULL) = (lease_expires_at IS NULL)",
    ),
)


def upgrade() -> None:
    op.add_column(
        "source_items",
        sa.Column("change_generation", sa.Integer(), server_default=sa.text("0"), nullable=False),
        schema=SCHEMA,
    )
    for table, name, condition in _CHECKS:
        op.create_check_constraint(name, table, condition, schema=SCHEMA)


def downgrade() -> None:
    for table, name, _ in reversed(_CHECKS):
        op.drop_constraint(name, table, type_="check", schema=SCHEMA)
    op.drop_column("source_items", "change_generation", schema=SCHEMA)
