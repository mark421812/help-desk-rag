from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Iterator
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Connection, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from helpdesk.domain.access import PermissionEffect, PrincipalType
from helpdesk.domain.conversations import ConversationStatus, TurnOutcome, TurnState
from helpdesk.domain.documents import Confidentiality, DocumentStatus, IndexingStatus
from helpdesk.domain.ingestion import IngestionJobStatus, IngestionJobType, SourceItemStatus
from helpdesk.infrastructure.persistence.models import SCHEMA, Base

pytestmark = pytest.mark.integration

T = TypeVar("T")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABLES = {
    "source_items",
    "ingestion_jobs",
    "documents",
    "document_access_rules",
    "document_versions",
    "chunks",
    "conversations",
    "turns",
    "last_full_scan",
}
VOCABULARIES: dict[str, type[StrEnum]] = {
    "ck_documents_status": DocumentStatus,
    "ck_documents_confidentiality": Confidentiality,
    "ck_document_access_rules_principal_type": PrincipalType,
    "ck_document_access_rules_effect": PermissionEffect,
    "ck_document_versions_indexing_status": IndexingStatus,
    "ck_chunks_indexing_status": IndexingStatus,
    "ck_conversations_status": ConversationStatus,
    "ck_turns_state": TurnState,
    "ck_turns_outcome": TurnOutcome,
    "ck_source_items_status": SourceItemStatus,
    "ck_ingestion_jobs_status": IngestionJobStatus,
    "ck_ingestion_jobs_job_type": IngestionJobType,
}


@pytest.fixture(scope="module")
def database_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="asyncpg") as postgres:
        yield postgres.get_connection_url()


@pytest.fixture
def alembic_config(database_url: str, monkeypatch: pytest.MonkeyPatch) -> Config:
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    return Config(str(PROJECT_ROOT / "alembic.ini"))


@pytest.fixture
def migrated(alembic_config: Config) -> Iterator[None]:
    command.upgrade(alembic_config, "head")
    yield
    command.downgrade(alembic_config, "base")


def run(database_url: str, work: Callable[[AsyncConnection], Awaitable[T]]) -> T:
    async def go() -> T:
        engine = create_async_engine(database_url)
        try:
            async with engine.connect() as connection:
                return await work(connection)
        finally:
            await engine.dispose()

    return asyncio.run(go())


async def tables(connection: AsyncConnection) -> set[str]:
    rows = await connection.execute(
        text("SELECT tablename FROM pg_tables WHERE schemaname = :schema"), {"schema": SCHEMA}
    )
    return set(rows.scalars())


async def model_diff(connection: AsyncConnection) -> list[Any]:
    def compare(sync_connection: Connection) -> list[Any]:
        context = MigrationContext.configure(
            sync_connection,
            opts={
                "include_schemas": True,
                "version_table_schema": SCHEMA,
                "compare_type": True,
                "compare_server_default": True,
            },
        )
        return list(compare_metadata(context, Base.metadata))

    return await connection.run_sync(compare)


async def vocabularies(connection: AsyncConnection) -> dict[str, set[str]]:
    rows = await connection.execute(
        text(
            "SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint "
            "WHERE connamespace = CAST(:schema AS regnamespace) AND contype = 'c'"
        ),
        {"schema": SCHEMA},
    )
    found = {str(name): str(definition) for name, definition in rows}
    return {name: set(re.findall(r"'([^']*)'::", found[name])) for name in VOCABULARIES}


def test_upgrade_matches_the_models_and_the_domain_vocabularies(
    migrated: None, database_url: str
) -> None:
    assert run(database_url, tables) == TABLES | {"alembic_version"}
    assert run(database_url, model_diff) == []
    assert run(database_url, vocabularies) == {
        name: {member.value for member in enum} for name, enum in VOCABULARIES.items()
    }


async def _execute(connection: AsyncConnection, sql: str, **params: object) -> None:
    async with connection.begin():
        await connection.execute(text(sql), params)


async def document_graph(connection: AsyncConnection) -> dict[str, object]:
    item, other_item = uuid4(), uuid4()
    document, other_document = uuid4(), uuid4()
    version, other_version = uuid4(), uuid4()
    for source_item, path in ((item, "a.pdf"), (other_item, "b.pdf")):
        await _execute(
            connection,
            "INSERT INTO helpdesk.source_items (id, relative_path, source_version, "
            "metadata_source_hash, file_size, modified_at, status) "
            "VALUES (:id, :path, 'mtime:1;size:1', :hash, 1, now(), 'completed')",
            id=source_item,
            path=path,
            hash="a" * 64,
        )
    for doc, source_item in ((document, item), (other_document, other_item)):
        await _execute(
            connection,
            "INSERT INTO helpdesk.documents (id, source_item_id, title, document_type, "
            "language, status, confidentiality, external_llm_allowed) "
            "VALUES (:id, :item, '簡章', 'general', 'zh-TW', 'draft', 'internal', false)",
            id=doc,
            item=source_item,
        )
    for ver, doc in ((version, document), (other_version, other_document)):
        await _execute(
            connection,
            "INSERT INTO helpdesk.document_versions (id, document_id, content_hash, mime_type, "
            "file_size, parser_name, parser_version, embedding_model, embedding_model_revision, "
            "embedding_dimension, chunk_count, indexing_status) VALUES (:id, :doc, :hash, "
            "'application/pdf', 1, 'pypdf', '1', 'e5', 'rev', 384, 1, 'indexed')",
            id=ver,
            doc=doc,
            hash="b" * 64,
        )
    await _execute(
        connection,
        "INSERT INTO helpdesk.chunks (id, document_id, version_id, chunk_index, content, "
        "content_hash, token_count, qdrant_point_id, indexing_status) "
        "VALUES (:id, :doc, :ver, 0, '內容', :hash, 3, :id, 'indexed')",
        id=uuid4(),
        doc=document,
        ver=version,
        hash="c" * 64,
    )
    await _execute(
        connection,
        "UPDATE helpdesk.documents SET current_version_id = :ver WHERE id = :doc",
        ver=version,
        doc=document,
    )

    rejected: list[str] = []
    for label, sql, params in (
        (
            "current version of another document",
            "UPDATE helpdesk.documents SET current_version_id = :ver WHERE id = :doc",
            {"ver": other_version, "doc": document},
        ),
        (
            "chunk whose version belongs to another document",
            "INSERT INTO helpdesk.chunks (id, document_id, version_id, chunk_index, content, "
            "content_hash, token_count, indexing_status) "
            "VALUES (:id, :doc, :ver, 1, '內容', :hash, 3, 'pending')",
            {"id": uuid4(), "doc": document, "ver": other_version, "hash": "c" * 64},
        ),
        (
            "indexed chunk without a Qdrant point",
            "INSERT INTO helpdesk.chunks (id, document_id, version_id, chunk_index, content, "
            "content_hash, token_count, indexing_status) "
            "VALUES (:id, :doc, :ver, 1, '內容', :hash, 3, 'indexed')",
            {"id": uuid4(), "doc": document, "ver": version, "hash": "c" * 64},
        ),
    ):
        try:
            await _execute(connection, sql, **params)
        except IntegrityError:
            rejected.append(label)

    await _execute(connection, "DELETE FROM helpdesk.source_items WHERE id = :id", id=item)
    remaining = await connection.execute(
        text(
            "SELECT (SELECT count(*) FROM helpdesk.documents), "
            "(SELECT count(*) FROM helpdesk.document_versions), "
            "(SELECT count(*) FROM helpdesk.chunks)"
        )
    )
    counts = tuple(remaining.one())
    await connection.rollback()
    return {"rejected": rejected, "remaining": counts}


def test_documents_point_only_at_their_own_versions_and_cascade(
    migrated: None, database_url: str
) -> None:
    assert run(database_url, document_graph) == {
        "rejected": [
            "current version of another document",
            "chunk whose version belongs to another document",
            "indexed chunk without a Qdrant point",
        ],
        "remaining": (1, 1, 0),
    }


async def turn_row_checks(connection: AsyncConnection) -> list[str]:
    conversation = uuid4()
    await _execute(
        connection,
        "INSERT INTO helpdesk.conversations (id, owner_id, status) "
        "VALUES (:id, 'anon-7f3a', 'active')",
        id=conversation,
    )
    insert_turn = (
        "INSERT INTO helpdesk.turns (id, conversation_id, idempotency_key, request_fingerprint, "
        "original_question, state, outcome, answer, error_code, retry_after_seconds) VALUES "
        "(:id, :conversation, :key, :fingerprint, '報名截止日？', :state, :outcome, "
        "CAST(:answer AS jsonb), :error, :retry_after)"
    )
    accepted: list[str] = []
    for label, state, outcome, answer, error, retry_after in (
        ("accepted turn", "accepted", None, None, None, None),
        ("completed without outcome", "completed", None, None, None, None),
        ("failed without error code", "failed", None, None, None, None),
        ("answered without answer", "completed", "answered", None, None, None),
        ("answer on another outcome", "completed", "insufficient_evidence", "[]", None, None),
        ("answered turn", "completed", "answered", "[]", None, None),
        ("failed turn", "failed_unknown", None, None, "processing_interrupted", None),
        ("rate-limited turn", "failed", None, None, "provider_rate_limited", 17),
        ("retry-after on an unknown failure", "failed_unknown", None, None, "timeout", 17),
        ("negative retry-after", "failed", None, None, "provider_rate_limited", -1),
    ):
        try:
            await _execute(
                connection,
                insert_turn,
                id=uuid4(),
                conversation=conversation,
                key=uuid4(),
                fingerprint="d" * 64,
                state=state,
                outcome=outcome,
                answer=answer,
                error=error,
                retry_after=retry_after,
            )
        except IntegrityError:
            continue
        accepted.append(label)
    return accepted


def test_turn_rows_keep_state_outcome_and_error_consistent(
    migrated: None, database_url: str
) -> None:
    assert run(database_url, turn_row_checks) == [
        "accepted turn",
        "answered turn",
        "failed turn",
        "rate-limited turn",
    ]


def test_downgrade_removes_every_table_and_upgrade_runs_again(
    alembic_config: Config, database_url: str
) -> None:
    command.upgrade(alembic_config, "head")
    command.downgrade(alembic_config, "base")
    assert run(database_url, tables) == {"alembic_version"}

    command.upgrade(alembic_config, "head")
    assert run(database_url, tables) == TABLES | {"alembic_version"}
    command.downgrade(alembic_config, "base")
