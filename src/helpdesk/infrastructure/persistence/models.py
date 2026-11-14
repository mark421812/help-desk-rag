"""`helpdesk` schema 的 SQLAlchemy model。

這些 class 只描述資料表形狀，不是 domain 型別；repository 負責兩者之間的轉換，ORM
session 因此不會滲進 domain。狀態欄位的 CHECK 約束寫在 migration 裡，整合測試會比對它們與
domain enum 是否一致。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Final
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SCHEMA: Final = "helpdesk"


class Base(DeclarativeBase):
    metadata = MetaData(schema=SCHEMA)


def _server_now() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


class SourceItemModel(Base):
    """文件根目錄下被掃描到的一個檔案，以及它在 ingestion 流程中的狀態。"""

    __tablename__ = "source_items"
    __table_args__ = (
        UniqueConstraint("relative_path", name="uq_source_items_relative_path"),
        Index("ix_source_items_status", "status"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    relative_path: Mapped[str] = mapped_column(Text)
    source_version: Mapped[str] = mapped_column(String(255))
    metadata_source_hash: Mapped[str] = mapped_column(String(64))
    mime_type: Mapped[str | None] = mapped_column(String(255))
    file_size: Mapped[int] = mapped_column(BigInteger)
    modified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(40))
    missing_scan_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    change_generation: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_successful_content_hash: Mapped[str | None] = mapped_column(String(64))
    last_successful_metadata_hash: Mapped[str | None] = mapped_column(String(64))
    error_code: Mapped[str | None] = mapped_column(String(255))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _server_now()
    updated_at: Mapped[datetime] = _server_now()
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class IngestionJobModel(Base):
    """PostgreSQL 當工作佇列；lease 欄位讓多個 worker 不會同時處理同一個 job。"""

    __tablename__ = "ingestion_jobs"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_ingestion_jobs_idempotency_key"),
        Index("ix_ingestion_jobs_claim", "status", "available_at", "created_at"),
        Index("ix_ingestion_jobs_source_item", "source_item_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    source_item_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            f"{SCHEMA}.source_items.id",
            ondelete="CASCADE",
            name="fk_ingestion_jobs_source_item",
        ),
    )
    job_type: Mapped[str] = mapped_column(String(30))
    idempotency_key: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(30))
    attempt_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    max_attempts: Mapped[int] = mapped_column(Integer, server_default=text("5"))
    available_at: Mapped[datetime] = _server_now()
    leased_by: Mapped[str | None] = mapped_column(String(255))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(255))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _server_now()
    updated_at: Mapped[datetime] = _server_now()
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class LastFullScanModel(Base):
    """最後一次完成的完整掃描，只有一列。排程靠它判斷啟動時要不要補掃。"""

    __tablename__ = "last_full_scan"

    singleton: Mapped[bool] = mapped_column(Boolean, primary_key=True, server_default=text("true"))
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class DocumentModel(Base):
    """文件與它的 metadata。`current_version_id` 只能指向這份文件自己的版本。"""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("source_item_id", name="uq_documents_source_item"),
        ForeignKeyConstraint(
            ["id", "current_version_id"],
            [f"{SCHEMA}.document_versions.document_id", f"{SCHEMA}.document_versions.id"],
            name="fk_documents_current_version",
            use_alter=True,
        ),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    source_item_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            f"{SCHEMA}.source_items.id",
            ondelete="CASCADE",
            name="fk_documents_source_item",
        ),
    )
    title: Mapped[str] = mapped_column(String(1000))
    document_type: Mapped[str] = mapped_column(String(100))
    language: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20))
    confidentiality: Mapped[str] = mapped_column(String(20))
    external_llm_allowed: Mapped[bool] = mapped_column(Boolean)
    effective_at: Mapped[date | None] = mapped_column(Date)
    expires_at: Mapped[date | None] = mapped_column(Date)
    current_version_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    created_at: Mapped[datetime] = _server_now()
    updated_at: Mapped[datetime] = _server_now()


class DocumentAccessRuleModel(Base):
    __tablename__ = "document_access_rules"

    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            f"{SCHEMA}.documents.id",
            ondelete="CASCADE",
            name="fk_document_access_rules_document",
        ),
        primary_key=True,
    )
    principal_type: Mapped[str] = mapped_column(String(20), primary_key=True)
    principal_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    effect: Mapped[str] = mapped_column(String(20), primary_key=True)


class DocumentVersionModel(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "content_hash",
            "embedding_model",
            "embedding_model_revision",
            name="uq_document_versions_fingerprint",
        ),
        UniqueConstraint("document_id", "id", name="uq_document_versions_document_id_id"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    document_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            f"{SCHEMA}.documents.id",
            ondelete="CASCADE",
            name="fk_document_versions_document",
        ),
    )
    content_hash: Mapped[str] = mapped_column(String(64))
    mime_type: Mapped[str] = mapped_column(String(255))
    file_size: Mapped[int] = mapped_column(BigInteger)
    parser_name: Mapped[str] = mapped_column(String(255))
    parser_version: Mapped[str] = mapped_column(String(255))
    embedding_model: Mapped[str] = mapped_column(String(255))
    embedding_model_revision: Mapped[str] = mapped_column(String(255))
    embedding_dimension: Mapped[int] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer)
    indexing_status: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = _server_now()


class ChunkModel(Base):
    """chunk 全文與引用定位。`(document_id, version_id)` 一起指向版本，兩者不會對不上。"""

    __tablename__ = "chunks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "version_id"],
            [f"{SCHEMA}.document_versions.document_id", f"{SCHEMA}.document_versions.id"],
            ondelete="CASCADE",
            name="fk_chunks_version",
        ),
        UniqueConstraint("version_id", "chunk_index", name="uq_chunks_version_index"),
        Index("ix_chunks_document", "document_id"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    document_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True))
    version_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True))
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    section_path: Mapped[str | None] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer)
    qdrant_point_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True))
    indexing_status: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = _server_now()


class ConversationModel(Base):
    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_owner_recent", "owner_id", "updated_at", "id"),)

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    owner_id: Mapped[str] = mapped_column(String(512))
    title: Mapped[str | None] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = _server_now()
    updated_at: Mapped[datetime] = _server_now()


class TurnModel(Base):
    """一個回合。`answer` 存回答段落，`citations` 存要顯示的引用。"""

    __tablename__ = "turns"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id", "idempotency_key", name="uq_turns_conversation_idempotency_key"
        ),
        Index("ix_turns_conversation_created", "conversation_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    conversation_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            f"{SCHEMA}.conversations.id",
            ondelete="CASCADE",
            name="fk_turns_conversation",
        ),
    )
    idempotency_key: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True))
    request_fingerprint: Mapped[str] = mapped_column(String(64))
    original_question: Mapped[str] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(20))
    standalone_question: Mapped[str | None] = mapped_column(Text)
    state: Mapped[str] = mapped_column(String(20))
    outcome: Mapped[str | None] = mapped_column(String(40))
    # 沒有回答時存 SQL NULL，不是 JSON null，CHECK `ck_turns_answer_only_when_answered` 才成立。
    answer: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB(none_as_null=True))
    citations: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, server_default=text("'[]'::jsonb")
    )
    error_code: Mapped[str | None] = mapped_column(String(100))
    retry_after_seconds: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = _server_now()
    updated_at: Mapped[datetime] = _server_now()
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
