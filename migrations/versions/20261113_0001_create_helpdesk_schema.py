"""Create the helpdesk schema tables for documents, ingestion and conversations.

Revision ID: 20261113_0001
Revises:
Create Date: 2026-11-13

狀態欄位的允許值直接寫成字面值，讓這個 revision 永遠產生同樣的 DDL；domain enum 之後
若有變動要另開 revision。整合測試會比對這些值與目前的 domain enum。
"""

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20261113_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "helpdesk"


def _one_of(column: str, values: Sequence[str]) -> str:
    return f"{column} IN ({', '.join(repr(value) for value in values)})"


def _uuid(name: str, *, nullable: bool = False, primary_key: bool = False) -> sa.Column[Any]:
    return sa.Column(
        name, postgresql.UUID(as_uuid=True), nullable=nullable, primary_key=primary_key
    )


def _now(name: str) -> sa.Column[Any]:
    return sa.Column(
        name, sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
    )


def _timestamp(name: str) -> sa.Column[Any]:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=True)


def upgrade() -> None:
    op.create_table(
        "source_items",
        _uuid("id", primary_key=True),
        sa.Column("relative_path", sa.Text(), nullable=False),
        sa.Column("source_version", sa.String(255), nullable=False),
        sa.Column("metadata_source_hash", sa.String(64), nullable=False),
        sa.Column("mime_type", sa.String(255), nullable=True),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("modified_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("missing_scan_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        _timestamp("last_seen_at"),
        sa.Column("last_successful_content_hash", sa.String(64), nullable=True),
        sa.Column("last_successful_metadata_hash", sa.String(64), nullable=True),
        sa.Column("error_code", sa.String(255), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        _now("created_at"),
        _now("updated_at"),
        _timestamp("completed_at"),
        _timestamp("deleted_at"),
        sa.UniqueConstraint("relative_path", name="uq_source_items_relative_path"),
        sa.CheckConstraint(
            "file_size >= 0 AND missing_scan_count >= 0", name="ck_source_items_counters"
        ),
        schema=SCHEMA,
    )
    op.create_index("ix_source_items_status", "source_items", ["status"], schema=SCHEMA)

    op.create_table(
        "ingestion_jobs",
        _uuid("id", primary_key=True),
        _uuid("source_item_id"),
        sa.Column("job_type", sa.String(30), nullable=False),
        sa.Column("idempotency_key", sa.String(512), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("attempt_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("max_attempts", sa.Integer(), server_default=sa.text("5"), nullable=False),
        _now("available_at"),
        sa.Column("leased_by", sa.String(255), nullable=True),
        _timestamp("lease_expires_at"),
        sa.Column("error_code", sa.String(255), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        _now("created_at"),
        _now("updated_at"),
        _timestamp("completed_at"),
        sa.ForeignKeyConstraint(
            ["source_item_id"],
            [f"{SCHEMA}.source_items.id"],
            ondelete="CASCADE",
            name="fk_ingestion_jobs_source_item",
        ),
        sa.UniqueConstraint("idempotency_key", name="uq_ingestion_jobs_idempotency_key"),
        sa.CheckConstraint(
            "attempt_count >= 0 AND max_attempts > 0", name="ck_ingestion_jobs_attempts"
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_ingestion_jobs_claim",
        "ingestion_jobs",
        ["status", "available_at", "created_at"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_ingestion_jobs_source_item",
        "ingestion_jobs",
        ["source_item_id", "created_at"],
        schema=SCHEMA,
    )

    op.create_table(
        "documents",
        _uuid("id", primary_key=True),
        _uuid("source_item_id"),
        sa.Column("title", sa.String(1000), nullable=False),
        sa.Column("document_type", sa.String(100), nullable=False),
        sa.Column("language", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("confidentiality", sa.String(20), nullable=False),
        sa.Column("external_llm_allowed", sa.Boolean(), nullable=False),
        sa.Column("effective_at", sa.Date(), nullable=True),
        sa.Column("expires_at", sa.Date(), nullable=True),
        _uuid("current_version_id", nullable=True),
        _now("created_at"),
        _now("updated_at"),
        sa.ForeignKeyConstraint(
            ["source_item_id"],
            [f"{SCHEMA}.source_items.id"],
            ondelete="CASCADE",
            name="fk_documents_source_item",
        ),
        sa.UniqueConstraint("source_item_id", name="uq_documents_source_item"),
        sa.CheckConstraint(
            _one_of("status", ("draft", "reviewing", "published", "expired", "archived")),
            name="ck_documents_status",
        ),
        sa.CheckConstraint(
            _one_of("confidentiality", ("public", "internal", "confidential", "restricted")),
            name="ck_documents_confidentiality",
        ),
        sa.CheckConstraint(
            "effective_at IS NULL OR expires_at IS NULL OR expires_at >= effective_at",
            name="ck_documents_effective_window",
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "document_access_rules",
        _uuid("document_id", primary_key=True),
        sa.Column("principal_type", sa.String(20), primary_key=True),
        sa.Column("principal_id", sa.String(512), primary_key=True),
        sa.Column("effect", sa.String(20), primary_key=True),
        sa.ForeignKeyConstraint(
            ["document_id"],
            [f"{SCHEMA}.documents.id"],
            ondelete="CASCADE",
            name="fk_document_access_rules_document",
        ),
        sa.CheckConstraint(
            _one_of("principal_type", ("user", "group")),
            name="ck_document_access_rules_principal_type",
        ),
        sa.CheckConstraint(
            _one_of("effect", ("allow", "deny")), name="ck_document_access_rules_effect"
        ),
        schema=SCHEMA,
    )

    op.create_table(
        "document_versions",
        _uuid("id", primary_key=True),
        _uuid("document_id"),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("mime_type", sa.String(255), nullable=False),
        sa.Column("file_size", sa.BigInteger(), nullable=False),
        sa.Column("parser_name", sa.String(255), nullable=False),
        sa.Column("parser_version", sa.String(255), nullable=False),
        sa.Column("embedding_model", sa.String(255), nullable=False),
        sa.Column("embedding_model_revision", sa.String(255), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("indexing_status", sa.String(20), nullable=False),
        _now("created_at"),
        sa.ForeignKeyConstraint(
            ["document_id"],
            [f"{SCHEMA}.documents.id"],
            ondelete="CASCADE",
            name="fk_document_versions_document",
        ),
        sa.UniqueConstraint(
            "document_id",
            "content_hash",
            "embedding_model",
            "embedding_model_revision",
            name="uq_document_versions_fingerprint",
        ),
        sa.UniqueConstraint("document_id", "id", name="uq_document_versions_document_id_id"),
        sa.CheckConstraint(
            _one_of("indexing_status", ("pending", "indexed")),
            name="ck_document_versions_indexing_status",
        ),
        sa.CheckConstraint(
            "file_size >= 0 AND embedding_dimension > 0 AND chunk_count > 0",
            name="ck_document_versions_counts",
        ),
        schema=SCHEMA,
    )
    # documents 與 document_versions 互相參照，所以這條 FK 只能在兩張表都存在後才加上。
    op.create_foreign_key(
        "fk_documents_current_version",
        "documents",
        "document_versions",
        ["id", "current_version_id"],
        ["document_id", "id"],
        source_schema=SCHEMA,
        referent_schema=SCHEMA,
    )

    op.create_table(
        "chunks",
        _uuid("id", primary_key=True),
        _uuid("document_id"),
        _uuid("version_id"),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("section_path", sa.Text(), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=False),
        _uuid("qdrant_point_id", nullable=True),
        sa.Column("indexing_status", sa.String(20), nullable=False),
        _now("created_at"),
        sa.ForeignKeyConstraint(
            ["document_id", "version_id"],
            [f"{SCHEMA}.document_versions.document_id", f"{SCHEMA}.document_versions.id"],
            ondelete="CASCADE",
            name="fk_chunks_version",
        ),
        sa.UniqueConstraint("version_id", "chunk_index", name="uq_chunks_version_index"),
        sa.CheckConstraint(
            _one_of("indexing_status", ("pending", "indexed")),
            name="ck_chunks_indexing_status",
        ),
        sa.CheckConstraint(
            "(indexing_status = 'indexed') = (qdrant_point_id IS NOT NULL)",
            name="ck_chunks_point_when_indexed",
        ),
        sa.CheckConstraint(
            "chunk_index >= 0 AND token_count > 0 AND (page_number IS NULL OR page_number > 0)",
            name="ck_chunks_positions",
        ),
        schema=SCHEMA,
    )
    op.create_index("ix_chunks_document", "chunks", ["document_id"], schema=SCHEMA)

    op.create_table(
        "conversations",
        _uuid("id", primary_key=True),
        sa.Column("owner_id", sa.String(512), nullable=False),
        sa.Column("title", sa.String(200), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        _now("created_at"),
        _now("updated_at"),
        sa.CheckConstraint(
            _one_of("status", ("active", "archived")), name="ck_conversations_status"
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_conversations_owner_recent",
        "conversations",
        ["owner_id", "updated_at", "id"],
        schema=SCHEMA,
    )

    op.create_table(
        "turns",
        _uuid("id", primary_key=True),
        _uuid("conversation_id"),
        _uuid("idempotency_key"),
        sa.Column("request_fingerprint", sa.String(64), nullable=False),
        sa.Column("original_question", sa.Text(), nullable=False),
        sa.Column("language", sa.String(20), nullable=True),
        sa.Column("standalone_question", sa.Text(), nullable=True),
        sa.Column("state", sa.String(20), nullable=False),
        sa.Column("outcome", sa.String(40), nullable=True),
        sa.Column("answer", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "citations",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("error_code", sa.String(100), nullable=True),
        _now("created_at"),
        _now("updated_at"),
        _timestamp("completed_at"),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            [f"{SCHEMA}.conversations.id"],
            ondelete="CASCADE",
            name="fk_turns_conversation",
        ),
        sa.UniqueConstraint(
            "conversation_id", "idempotency_key", name="uq_turns_conversation_idempotency_key"
        ),
        sa.CheckConstraint(
            _one_of(
                "state",
                ("accepted", "retrieving", "generating", "completed", "failed", "failed_unknown"),
            ),
            name="ck_turns_state",
        ),
        sa.CheckConstraint(
            "outcome IS NULL OR "
            + _one_of(
                "outcome",
                (
                    "answered",
                    "unable_to_answer",
                    "insufficient_evidence",
                    "external_generation_blocked",
                    "outbound_policy_blocked",
                ),
            ),
            name="ck_turns_outcome",
        ),
        sa.CheckConstraint(
            "(state = 'completed') = (outcome IS NOT NULL)",
            name="ck_turns_completed_has_outcome",
        ),
        sa.CheckConstraint(
            "(state IN ('failed', 'failed_unknown')) = (error_code IS NOT NULL)",
            name="ck_turns_failed_has_error_code",
        ),
        sa.CheckConstraint(
            "(outcome IS NOT DISTINCT FROM 'answered') = (answer IS NOT NULL)",
            name="ck_turns_answer_only_when_answered",
        ),
        sa.CheckConstraint(
            "request_fingerprint ~ '^[0-9a-f]{64}$'", name="ck_turns_request_fingerprint"
        ),
        sa.CheckConstraint(
            "char_length(original_question) BETWEEN 1 AND 20000",
            name="ck_turns_original_question_length",
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_turns_conversation_created",
        "turns",
        ["conversation_id", "created_at"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_table("turns", schema=SCHEMA)
    op.drop_table("conversations", schema=SCHEMA)
    op.drop_table("chunks", schema=SCHEMA)
    op.drop_constraint(
        "fk_documents_current_version", "documents", type_="foreignkey", schema=SCHEMA
    )
    op.drop_table("document_versions", schema=SCHEMA)
    op.drop_table("document_access_rules", schema=SCHEMA)
    op.drop_table("documents", schema=SCHEMA)
    op.drop_table("ingestion_jobs", schema=SCHEMA)
    op.drop_table("source_items", schema=SCHEMA)
