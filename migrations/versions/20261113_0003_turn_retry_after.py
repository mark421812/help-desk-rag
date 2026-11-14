"""Record how long a rate-limited turn asks the user to wait.

Revision ID: 20261113_0003
Revises: 20261113_0002
Create Date: 2026-11-13

provider 回 429 時回合確定失敗，`retry_after_seconds` 記下 provider 要求的等待秒數，網頁據此
顯示「請稍候再試」。只有 `failed` 的回合可以帶這個值，和 domain 的 invariant 一致。
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261113_0003"
down_revision: str | None = "20261113_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "helpdesk"


def upgrade() -> None:
    op.add_column(
        "turns", sa.Column("retry_after_seconds", sa.Integer(), nullable=True), schema=SCHEMA
    )
    op.create_check_constraint(
        "ck_turns_retry_after",
        "turns",
        "retry_after_seconds IS NULL OR (state = 'failed' AND retry_after_seconds >= 0)",
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_constraint("ck_turns_retry_after", "turns", type_="check", schema=SCHEMA)
    op.drop_column("turns", "retry_after_seconds", schema=SCHEMA)
