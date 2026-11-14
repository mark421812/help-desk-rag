"""Alembic 執行環境：只管 `helpdesk` schema。

資料庫 URL 從 `Settings` 讀取，與兩個程序用同一份設定與驗證。版本表放在
`helpdesk.alembic_version`。autogenerate 只比對 `helpdesk` schema，同一個資料庫裡的其他
schema 不會被當成要刪除的差異。
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection, pool, text
from sqlalchemy.ext.asyncio import create_async_engine

from helpdesk.config import Settings
from helpdesk.infrastructure.persistence.models import SCHEMA, Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)


def _only_helpdesk_schema(name: str | None, type_: str, parent_names: object) -> bool:
    return type_ != "schema" or name == SCHEMA


def _run_migrations(connection: Connection) -> None:
    # 版本表就在這個 schema 裡，所以 schema 必須在 Alembic 建版本表之前存在。
    connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"'))
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        version_table_schema=SCHEMA,
        include_schemas=True,
        include_name=_only_helpdesk_schema,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def _run_online() -> None:
    engine = create_async_engine(Settings().database_url, poolclass=pool.NullPool)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(_run_migrations)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    raise RuntimeError("helpdesk migrations only run against a live database")
asyncio.run(_run_online())
