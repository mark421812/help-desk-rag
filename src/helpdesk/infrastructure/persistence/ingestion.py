"""ingestion 的 PostgreSQL repository 與 unit of work。

讀取一律帶 `populate_existing`，寫入用 INSERT … ON CONFLICT，ORM 的 identity map 因此不會
留下舊值。寫入用的讀取會 `FOR UPDATE` 鎖列，worker 用 `SKIP LOCKED` 取 job，同一個檔案
不會同時被兩個 worker 處理。
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from types import TracebackType
from typing import Any, Final, Self, TypeVarTuple

from sqlalchemy import Select, and_, delete, exists, func, or_, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased

from helpdesk.domain.access import AccessRule, PermissionEffect, PrincipalType
from helpdesk.domain.documents import (
    Chunk,
    ChunkId,
    Confidentiality,
    Document,
    DocumentId,
    DocumentMetadata,
    DocumentStatus,
    DocumentVersion,
    IndexingStatus,
    SourceItemId,
    VersionId,
)
from helpdesk.domain.ingestion import (
    IngestionJob,
    IngestionJobStatus,
    IngestionJobType,
    JobId,
    ScanInProgressError,
    SourceItem,
    SourceItemStatus,
)
from helpdesk.infrastructure.persistence.models import (
    ChunkModel,
    DocumentAccessRuleModel,
    DocumentModel,
    DocumentVersionModel,
    IngestionJobModel,
    LastFullScanModel,
    SourceItemModel,
)

_SCAN_LOCK_KEY: Final = int.from_bytes(
    hashlib.sha256(b"helpdesk:ingestion-scan").digest()[:8], "big", signed=True
)


_Columns = TypeVarTuple("_Columns")


def _fresh(statement: Select[*_Columns]) -> Select[*_Columns]:
    return statement.execution_options(populate_existing=True)


class SqlSourceItems:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, item_id: SourceItemId) -> SourceItem | None:
        row = await self._session.scalar(
            _fresh(select(SourceItemModel).where(SourceItemModel.id == item_id).with_for_update())
        )
        return None if row is None else _item(row)

    async def by_path(self, relative_path: str) -> SourceItem | None:
        row = await self._session.scalar(
            _fresh(
                select(SourceItemModel)
                .where(SourceItemModel.relative_path == relative_path)
                .with_for_update()
            )
        )
        return None if row is None else _item(row)

    async def all(self) -> list[SourceItem]:
        rows = await self._session.scalars(
            _fresh(
                select(SourceItemModel).order_by(SourceItemModel.relative_path).with_for_update()
            )
        )
        return [_item(row) for row in rows]

    async def save(self, item: SourceItem) -> None:
        values = _item_values(item)
        statement = insert(SourceItemModel).values(values)
        await self._session.execute(
            statement.on_conflict_do_update(
                index_elements=[SourceItemModel.id],
                set_={**{key: statement.excluded[key] for key in values}, "updated_at": func.now()},
            )
        )

    async def page(
        self, *, status: SourceItemStatus | None, offset: int, limit: int
    ) -> list[SourceItem]:
        statement = select(SourceItemModel).order_by(SourceItemModel.relative_path)
        if status is not None:
            statement = statement.where(SourceItemModel.status == status.value)
        rows = await self._session.scalars(_fresh(statement.offset(offset).limit(limit)))
        return [_item(row) for row in rows]


class SqlFullScans:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record_completed(self, at: datetime) -> None:
        statement = insert(LastFullScanModel).values(singleton=True, completed_at=at)
        await self._session.execute(
            statement.on_conflict_do_update(
                index_elements=[LastFullScanModel.singleton],
                set_={"completed_at": statement.excluded.completed_at},
            )
        )

    async def last_completed_at(self) -> datetime | None:
        return await self._session.scalar(select(LastFullScanModel.completed_at))


class SqlIngestionJobs:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, job_id: JobId) -> IngestionJob | None:
        row = await self._session.scalar(
            _fresh(
                select(IngestionJobModel).where(IngestionJobModel.id == job_id).with_for_update()
            )
        )
        return None if row is None else _job(row)

    async def add_if_absent(self, job: IngestionJob) -> bool:
        inserted = await self._session.scalar(
            insert(IngestionJobModel)
            .values(_job_values(job))
            .on_conflict_do_nothing()
            .returning(IngestionJobModel.id)
        )
        return inserted is not None

    async def save(self, job: IngestionJob) -> None:
        await self._session.execute(
            update(IngestionJobModel)
            .where(IngestionJobModel.id == job.id)
            .values(**_job_values(job), updated_at=func.now())
        )

    async def claim_next(
        self, worker_id: str, now: datetime, lease_seconds: float
    ) -> IngestionJob | None:
        sibling = aliased(IngestionJobModel)
        sibling_running = exists().where(
            sibling.source_item_id == IngestionJobModel.source_item_id,
            sibling.id != IngestionJobModel.id,
            sibling.status == IngestionJobStatus.PROCESSING.value,
            sibling.lease_expires_at > now,
        )
        row = await self._session.scalar(
            _fresh(
                select(IngestionJobModel)
                .join(SourceItemModel, SourceItemModel.id == IngestionJobModel.source_item_id)
                .where(
                    IngestionJobModel.available_at <= now,
                    ~sibling_running,
                    or_(
                        IngestionJobModel.status == IngestionJobStatus.PENDING.value,
                        and_(
                            IngestionJobModel.status == IngestionJobStatus.PROCESSING.value,
                            IngestionJobModel.lease_expires_at <= now,
                        ),
                    ),
                )
                .order_by(IngestionJobModel.available_at, IngestionJobModel.created_at)
                .with_for_update(of=(IngestionJobModel, SourceItemModel), skip_locked=True)
                .limit(1)
            )
        )
        if row is None:
            return None
        job = _job(row).claim(worker_id, now, timedelta(seconds=lease_seconds))
        await self.save(job)
        return job

    async def latest_by_item(
        self, item_ids: Sequence[SourceItemId]
    ) -> Mapping[SourceItemId, IngestionJob]:
        rows = await self._session.scalars(
            _fresh(
                select(IngestionJobModel)
                .where(IngestionJobModel.source_item_id.in_(item_ids))
                .order_by(IngestionJobModel.created_at, IngestionJobModel.updated_at)
            )
        )
        return {SourceItemId(row.source_item_id): _job(row) for row in rows}


class SqlDocuments:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def for_item(self, item_id: SourceItemId) -> Document | None:
        row = await self._session.scalar(
            _fresh(select(DocumentModel).where(DocumentModel.source_item_id == item_id))
        )
        if row is None:
            return None
        rules = await self._session.scalars(
            _fresh(
                select(DocumentAccessRuleModel)
                .where(DocumentAccessRuleModel.document_id == row.id)
                .order_by(
                    DocumentAccessRuleModel.principal_type,
                    DocumentAccessRuleModel.principal_id,
                    DocumentAccessRuleModel.effect,
                )
            )
        )
        return document_from_row(row, list(rules))

    async def save_metadata(self, document: Document) -> None:
        metadata = document.metadata
        values = {
            "title": metadata.title,
            "document_type": metadata.document_type,
            "language": metadata.language,
            "status": metadata.status.value,
            "confidentiality": metadata.confidentiality.value,
            "external_llm_allowed": metadata.external_llm_allowed,
            "effective_at": metadata.effective_at,
            "expires_at": metadata.expires_at,
        }
        await self._session.execute(
            insert(DocumentModel)
            .values(id=document.id, source_item_id=document.source_item_id, **values)
            .on_conflict_do_update(
                index_elements=[DocumentModel.id], set_={**values, "updated_at": func.now()}
            )
        )
        await self._session.execute(
            delete(DocumentAccessRuleModel).where(
                DocumentAccessRuleModel.document_id == document.id
            )
        )
        if metadata.access_rules:
            await self._session.execute(
                insert(DocumentAccessRuleModel).values(
                    [
                        {
                            "document_id": document.id,
                            "principal_type": rule.principal_type.value,
                            "principal_id": rule.principal_id,
                            "effect": rule.effect.value,
                        }
                        for rule in metadata.access_rules
                    ]
                )
            )

    async def version(
        self, document_id: DocumentId, content_hash: str, model_id: str, revision: str
    ) -> DocumentVersion | None:
        row = await self._session.scalar(
            _fresh(
                select(DocumentVersionModel).where(
                    DocumentVersionModel.document_id == document_id,
                    DocumentVersionModel.content_hash == content_hash,
                    DocumentVersionModel.embedding_model == model_id,
                    DocumentVersionModel.embedding_model_revision == revision,
                )
            )
        )
        return None if row is None else _version(row)

    async def add_version(self, version: DocumentVersion, chunks: Sequence[Chunk]) -> None:
        await self._session.execute(
            insert(DocumentVersionModel).values(
                id=version.id,
                document_id=version.document_id,
                content_hash=version.content_hash,
                mime_type=version.mime_type,
                file_size=version.file_size,
                parser_name=version.parser_name,
                parser_version=version.parser_version,
                embedding_model=version.embedding_model,
                embedding_model_revision=version.embedding_model_revision,
                embedding_dimension=version.embedding_dimension,
                chunk_count=version.chunk_count,
                indexing_status=version.status.value,
            )
        )
        await self._session.execute(
            insert(ChunkModel).values(
                [
                    {
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "version_id": chunk.version_id,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "content_hash": chunk.content_hash,
                        "section_path": chunk.section_path,
                        "page_number": chunk.page_number,
                        "token_count": chunk.token_count,
                        "qdrant_point_id": chunk.qdrant_point_id,
                        "indexing_status": chunk.status.value,
                    }
                    for chunk in chunks
                ]
            )
        )

    async def chunks(self, version_id: VersionId) -> list[Chunk]:
        rows = await self._session.scalars(
            _fresh(
                select(ChunkModel)
                .where(ChunkModel.version_id == version_id)
                .order_by(ChunkModel.chunk_index)
            )
        )
        return [chunk_from_row(row) for row in rows]

    async def activate(self, document_id: DocumentId, version_id: VersionId) -> None:
        indexed = IndexingStatus.INDEXED.value
        await self._session.execute(
            update(ChunkModel)
            .where(ChunkModel.version_id == version_id)
            .values(indexing_status=indexed, qdrant_point_id=ChunkModel.id)
        )
        await self._session.execute(
            update(DocumentVersionModel)
            .where(DocumentVersionModel.id == version_id)
            .values(indexing_status=indexed)
        )
        await self._session.execute(
            update(DocumentModel)
            .where(DocumentModel.id == document_id)
            .values(current_version_id=version_id, updated_at=func.now())
        )
        await self._session.execute(
            delete(DocumentVersionModel).where(
                DocumentVersionModel.document_id == document_id,
                DocumentVersionModel.id != version_id,
            )
        )

    async def delete(self, document_id: DocumentId) -> None:
        await self._session.execute(delete(DocumentModel).where(DocumentModel.id == document_id))


class SqlUnitOfWork:
    items: SqlSourceItems
    jobs: SqlIngestionJobs
    documents: SqlDocuments
    full_scans: SqlFullScans

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def __aenter__(self) -> Self:
        self._session = self._sessions()
        self.items = SqlSourceItems(self._session)
        self.jobs = SqlIngestionJobs(self._session)
        self.documents = SqlDocuments(self._session)
        self.full_scans = SqlFullScans(self._session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._session.close()

    async def commit(self) -> None:
        await self._session.commit()


class PostgresScanLock:
    """用 transaction 範圍的 advisory lock 讓同一時間只有一次掃描。"""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        async with self._engine.connect() as connection, connection.begin():
            acquired = await connection.scalar(
                text("SELECT pg_try_advisory_xact_lock(:key)"), {"key": _SCAN_LOCK_KEY}
            )
            if acquired is not True:
                raise ScanInProgressError("another scan is in progress")
            yield


def _item(row: SourceItemModel) -> SourceItem:
    return SourceItem(
        id=SourceItemId(row.id),
        relative_path=row.relative_path,
        source_version=row.source_version,
        metadata_source_hash=row.metadata_source_hash,
        file_size=row.file_size,
        modified_at=row.modified_at,
        status=SourceItemStatus(row.status),
        mime_type=row.mime_type,
        missing_scan_count=row.missing_scan_count,
        change_generation=row.change_generation,
        last_seen_at=row.last_seen_at,
        last_successful_content_hash=row.last_successful_content_hash,
        last_successful_metadata_hash=row.last_successful_metadata_hash,
        error_code=row.error_code,
        error_message=row.error_message,
        completed_at=row.completed_at,
        deleted_at=row.deleted_at,
    )


def _item_values(item: SourceItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "relative_path": item.relative_path,
        "source_version": item.source_version,
        "metadata_source_hash": item.metadata_source_hash,
        "mime_type": item.mime_type,
        "file_size": item.file_size,
        "modified_at": item.modified_at,
        "status": item.status.value,
        "missing_scan_count": item.missing_scan_count,
        "change_generation": item.change_generation,
        "last_seen_at": item.last_seen_at,
        "last_successful_content_hash": item.last_successful_content_hash,
        "last_successful_metadata_hash": item.last_successful_metadata_hash,
        "error_code": item.error_code,
        "error_message": item.error_message,
        "completed_at": item.completed_at,
        "deleted_at": item.deleted_at,
    }


def _job(row: IngestionJobModel) -> IngestionJob:
    return IngestionJob(
        id=JobId(row.id),
        source_item_id=SourceItemId(row.source_item_id),
        job_type=IngestionJobType(row.job_type),
        idempotency_key=row.idempotency_key,
        available_at=row.available_at,
        max_attempts=row.max_attempts,
        status=IngestionJobStatus(row.status),
        attempt_count=row.attempt_count,
        leased_by=row.leased_by,
        lease_expires_at=row.lease_expires_at,
        error_code=row.error_code,
        error_message=row.error_message,
        completed_at=row.completed_at,
    )


def _job_values(job: IngestionJob) -> dict[str, Any]:
    return {
        "id": job.id,
        "source_item_id": job.source_item_id,
        "job_type": job.job_type.value,
        "idempotency_key": job.idempotency_key,
        "status": job.status.value,
        "attempt_count": job.attempt_count,
        "max_attempts": job.max_attempts,
        "available_at": job.available_at,
        "leased_by": job.leased_by,
        "lease_expires_at": job.lease_expires_at,
        "error_code": job.error_code,
        "error_message": job.error_message,
        "completed_at": job.completed_at,
    }


def document_from_row(row: DocumentModel, rules: Sequence[DocumentAccessRuleModel]) -> Document:
    return Document(
        id=DocumentId(row.id),
        source_item_id=SourceItemId(row.source_item_id),
        metadata=DocumentMetadata(
            title=row.title,
            document_type=row.document_type,
            language=row.language,
            status=DocumentStatus(row.status),
            confidentiality=Confidentiality(row.confidentiality),
            external_llm_allowed=row.external_llm_allowed,
            effective_at=row.effective_at,
            expires_at=row.expires_at,
            access_rules=tuple(
                AccessRule(
                    principal_type=PrincipalType(rule.principal_type),
                    principal_id=rule.principal_id,
                    effect=PermissionEffect(rule.effect),
                )
                for rule in rules
            ),
        ),
        current_version_id=None
        if row.current_version_id is None
        else VersionId(row.current_version_id),
    )


def _version(row: DocumentVersionModel) -> DocumentVersion:
    return DocumentVersion(
        id=VersionId(row.id),
        document_id=DocumentId(row.document_id),
        content_hash=row.content_hash,
        mime_type=row.mime_type,
        file_size=row.file_size,
        parser_name=row.parser_name,
        parser_version=row.parser_version,
        embedding_model=row.embedding_model,
        embedding_model_revision=row.embedding_model_revision,
        embedding_dimension=row.embedding_dimension,
        chunk_count=row.chunk_count,
        status=IndexingStatus(row.indexing_status),
    )


def chunk_from_row(row: ChunkModel) -> Chunk:
    return Chunk(
        id=ChunkId(row.id),
        document_id=DocumentId(row.document_id),
        version_id=VersionId(row.version_id),
        chunk_index=row.chunk_index,
        content=row.content,
        token_count=row.token_count,
        section_path=row.section_path,
        page_number=row.page_number,
        status=IndexingStatus(row.indexing_status),
        qdrant_point_id=row.qdrant_point_id,
    )


def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)
