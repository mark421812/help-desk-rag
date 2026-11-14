"""取得並執行 ingestion job。

處理 job 的每一步都可以重跑：版本與 chunk 的 ID 由內容推導，Qdrant points 用 chunk ID
寫入，所以中途當掉後重試只會覆蓋同樣的資料。新版本先把所有 points 寫完，PostgreSQL 才把
它切成目前版本，最後 `VectorIndex.publish` 讓 Qdrant 跟上並刪掉舊版本的 points。
永久拒絕的文件會從索引撤下，ACL 開啟後缺 metadata 的文件因此不會留在檢索結果裡。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from uuid import NAMESPACE_URL, uuid5

from helpdesk.application.ingestion.ports import (
    ChunkDraft,
    Chunker,
    Clock,
    DiscoveredFile,
    DocumentParser,
    DocumentSource,
    MetadataPolicy,
    MetadataSource,
    ParsedDocument,
    PassageEmbedder,
    UnitOfWork,
    UnitOfWorkFactory,
    VectorIndex,
)
from helpdesk.domain.documents import (
    Chunk,
    ChunkId,
    Document,
    DocumentId,
    DocumentMetadata,
    DocumentVersion,
    SourceItemId,
    VersionId,
)
from helpdesk.domain.ingestion import (
    DocumentRejectedError,
    FileTooLargeError,
    IngestionJob,
    IngestionJobStatus,
    IngestionJobType,
    InvalidDocumentError,
    JobId,
    LeaseLostError,
    SourceChangedError,
    SourceItem,
    SourceItemStatus,
)


@dataclass(frozen=True, slots=True)
class ProcessingLimits:
    max_file_size_bytes: int
    retry_base_delay: timedelta
    max_document_characters: int = 20_000_000
    max_document_chunks: int = 10_000


@dataclass(frozen=True, slots=True)
class JobResult:
    job: IngestionJob
    item: SourceItem


def indexed_content_hash(parsed: ParsedDocument, chunks: Sequence[ChunkDraft]) -> str:
    """parser 身分與切塊結果的 canonical JSON SHA-256。"""

    payload = {
        "parser_name": parsed.parser_name,
        "parser_version": parsed.parser_version,
        "chunks": [
            {
                "content": chunk.content,
                "section_path": chunk.section_path,
                "page_number": chunk.page_number,
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def document_id_for(item_id: SourceItemId) -> DocumentId:
    return DocumentId(uuid5(NAMESPACE_URL, f"helpdesk:document:{item_id}"))


class ClaimNextJob:
    def __init__(self, uow_factory: UnitOfWorkFactory, lease: timedelta, clock: Clock) -> None:
        self._uow_factory = uow_factory
        self._lease = lease
        self._clock = clock

    async def run(self, worker_id: str) -> IngestionJob | None:
        async with self._uow_factory() as uow:
            job = await uow.jobs.claim_next(worker_id, self._clock(), self._lease.total_seconds())
            await uow.commit()
        return job


@dataclass(frozen=True, slots=True)
class _Prepared:
    file: DiscoveredFile
    parsed: ParsedDocument
    chunks: tuple[ChunkDraft, ...]
    metadata: DocumentMetadata
    content_hash: str
    dimension: int


class ProcessJob:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        *,
        source: DocumentSource,
        parser: DocumentParser,
        chunker: Chunker,
        embedder: PassageEmbedder,
        metadata_source: MetadataSource,
        metadata_policy: MetadataPolicy,
        index: VectorIndex,
        limits: ProcessingLimits,
        clock: Clock,
    ) -> None:
        self._uow_factory = uow_factory
        self._source = source
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._metadata_source = metadata_source
        self._metadata_policy = metadata_policy
        self._index = index
        self._limits = limits
        self._clock = clock

    async def run(self, job_id: JobId, worker_id: str) -> JobResult:
        """執行一個已取得 lease 的 job。只有 lease 遺失時丟出例外，其他失敗都記在 job 上。"""

        started = await self._begin(job_id, worker_id)
        if started.job.status is not IngestionJobStatus.PROCESSING:
            return started
        try:
            if started.job.job_type is IngestionJobType.DELETE_ITEM:
                return await self._delete(job_id, worker_id, started.item)
            return await self._process(job_id, worker_id, started.item)
        except LeaseLostError:
            raise
        except DocumentRejectedError as rejection:
            try:
                return await self._reject(job_id, worker_id, started.item, rejection)
            except LeaseLostError:
                raise
            except Exception as error:
                return await self._fail(job_id, worker_id, started.item, "withdraw_failed", error)
        except SourceChangedError as error:
            return await self._fail(job_id, worker_id, started.item, "source_changed", error)
        except Exception as error:
            return await self._fail(job_id, worker_id, started.item, "processing_error", error)

    async def _begin(self, job_id: JobId, worker_id: str) -> JobResult:
        now = self._clock()
        async with self._uow_factory() as uow:
            job, item = await _leased(uow, job_id, worker_id, now)
            deleting = job.job_type is IngestionJobType.DELETE_ITEM
            still_relevant = item.is_missing if deleting else not item.is_missing
            if item.status is SourceItemStatus.DELETED or not still_relevant:
                job = job.complete(worker_id, now)
                await uow.jobs.save(job)
            elif not deleting:
                item = item.begin_processing()
                await uow.items.save(item)
            await uow.commit()
        return JobResult(job=job, item=item)

    async def _process(self, job_id: JobId, worker_id: str, item: SourceItem) -> JobResult:
        prepared = await self._prepare(item)
        indexing = await self._stage(job_id, worker_id, item, prepared)
        if indexing is None:
            return await self._settle(job_id, worker_id, item, None)
        document, version, chunks = indexing
        await self._index.ensure_collection(prepared.dimension)
        if chunks:
            vectors = await asyncio.to_thread(
                self._embedder.embed, [chunk.content for chunk in chunks]
            )
            await self._index.upsert(document, chunks, vectors)
            async with self._uow_factory() as uow:
                await _leased(uow, job_id, worker_id, self._clock())
                await uow.documents.activate(document.id, version.id)
                await uow.commit()
            document = replace(document, current_version_id=version.id)
        await self._index.publish(document)

        def completed(current: SourceItem) -> SourceItem:
            return current.complete(
                content_hash=prepared.content_hash,
                mime_type=prepared.parsed.mime_type,
                at=self._clock(),
            )

        return await self._settle(job_id, worker_id, item, completed)

    async def _prepare(self, item: SourceItem) -> _Prepared:
        file = await self._unchanged_file(item)
        if file.file_size > self._limits.max_file_size_bytes:
            raise FileTooLargeError(
                f"file size {file.file_size} exceeds the limit {self._limits.max_file_size_bytes}"
            )
        declared = await asyncio.to_thread(self._metadata_source.declared, item.relative_path)
        metadata = self._metadata_policy.resolve(declared, item.relative_path)
        parsed = await asyncio.to_thread(self._parser.parse, file.path)
        chunks = await asyncio.to_thread(self._chunker.chunk, parsed)
        self._check_size(parsed, chunks)
        await self._unchanged_file(item)
        dimension = await asyncio.to_thread(self._embedder.dimension)
        return _Prepared(
            file=file,
            parsed=parsed,
            chunks=chunks,
            metadata=metadata,
            content_hash=indexed_content_hash(parsed, chunks),
            dimension=dimension,
        )

    async def _stage(
        self, job_id: JobId, worker_id: str, item: SourceItem, prepared: _Prepared
    ) -> tuple[Document, DocumentVersion, list[Chunk]] | None:
        """保存 metadata，並確保要索引的版本與 chunk 已在 PostgreSQL。

        回傳 None 代表掃描已經看到更新的檔案，這個 job 被取代。回傳空的 chunk 清單代表
        內容沒變，只需要把 metadata 同步到 Qdrant。
        """

        async with self._uow_factory() as uow:
            _, current = await _leased(uow, job_id, worker_id, self._clock())
            if current.discovery_fingerprint != item.discovery_fingerprint:
                return None
            existing = await uow.documents.for_item(item.id)
            document = Document(
                id=existing.id if existing else document_id_for(item.id),
                source_item_id=item.id,
                metadata=prepared.metadata,
                current_version_id=existing.current_version_id if existing else None,
            )
            await uow.documents.save_metadata(document)
            version = await uow.documents.version(
                document.id,
                prepared.content_hash,
                self._embedder.model_id,
                self._embedder.revision,
            )
            if version is None:
                version, chunks = self._new_version(document.id, prepared)
                await uow.documents.add_version(version, chunks)
            elif version.id == document.current_version_id:
                chunks = []
            else:
                chunks = await uow.documents.chunks(version.id)
            await uow.commit()
        return document, version, chunks

    def _new_version(
        self, document_id: DocumentId, prepared: _Prepared
    ) -> tuple[DocumentVersion, list[Chunk]]:
        model_id, revision = self._embedder.model_id, self._embedder.revision
        version_id = VersionId(
            uuid5(
                NAMESPACE_URL,
                f"helpdesk:version:{document_id}:{prepared.content_hash}:{model_id}:{revision}",
            )
        )
        chunks = [
            Chunk(
                id=ChunkId(uuid5(NAMESPACE_URL, f"helpdesk:chunk:{version_id}:{index}")),
                document_id=document_id,
                version_id=version_id,
                chunk_index=index,
                content=draft.content,
                token_count=draft.token_count,
                section_path=draft.section_path,
                page_number=draft.page_number,
            )
            for index, draft in enumerate(prepared.chunks)
        ]
        version = DocumentVersion(
            id=version_id,
            document_id=document_id,
            content_hash=prepared.content_hash,
            mime_type=prepared.parsed.mime_type,
            file_size=prepared.file.file_size,
            parser_name=prepared.parsed.parser_name,
            parser_version=prepared.parsed.parser_version,
            embedding_model=model_id,
            embedding_model_revision=revision,
            embedding_dimension=prepared.dimension,
            chunk_count=len(chunks),
        )
        return version, chunks

    async def _delete(self, job_id: JobId, worker_id: str, item: SourceItem) -> JobResult:
        document = await self._document_of(item)
        if document is not None:
            await self._index.delete_document(document.id)
        now = self._clock()
        async with self._uow_factory() as uow:
            job, current = await _leased(uow, job_id, worker_id, now)
            if document is not None:
                await uow.documents.delete(document.id)
            # 檔案在刪除途中又出現時，掃描已把它排回處理；文件照樣刪掉，讓新的 job 從頭建立。
            if current.is_missing:
                current = current.mark_deleted(now)
                await uow.items.save(current)
            job = job.complete(worker_id, now)
            await uow.jobs.save(job)
            await uow.commit()
        return JobResult(job=job, item=current)

    async def _reject(
        self,
        job_id: JobId,
        worker_id: str,
        item: SourceItem,
        rejection: DocumentRejectedError,
    ) -> JobResult:
        document = await self._document_of(item)
        if document is not None:
            await self._index.delete_document(document.id)
            async with self._uow_factory() as uow:
                await _leased(uow, job_id, worker_id, self._clock())
                await uow.documents.delete(document.id)
                await uow.commit()
        return await self._settle(
            job_id,
            worker_id,
            item,
            lambda current: current.reject(rejection),
            failure=(rejection.code, str(rejection), False),
        )

    async def _fail(
        self, job_id: JobId, worker_id: str, item: SourceItem, code: str, error: Exception
    ) -> JobResult:
        message = f"{type(error).__name__}: {error}"
        return await self._settle(
            job_id,
            worker_id,
            item,
            lambda current: current.fail(code, message),
            failure=(code, message, True),
        )

    async def _settle(
        self,
        job_id: JobId,
        worker_id: str,
        item: SourceItem,
        update: Callable[[SourceItem], SourceItem] | None,
        *,
        failure: tuple[str, str, bool] | None = None,
    ) -> JobResult:
        """寫回 job 結果。檔案在處理期間又被掃描改動時，item 交給較新的 job 處理。"""

        now = self._clock()
        async with self._uow_factory() as uow:
            job, current = await _leased(uow, job_id, worker_id, now)
            if (
                update is not None
                and current.status is SourceItemStatus.PROCESSING
                and current.discovery_fingerprint == item.discovery_fingerprint
            ):
                current = update(current)
                await uow.items.save(current)
            if failure is None:
                job = job.complete(worker_id, now)
            else:
                code, message, retryable = failure
                job = job.fail(
                    worker_id,
                    now,
                    code=code,
                    message=message,
                    retryable=retryable,
                    base_delay=self._limits.retry_base_delay,
                )
            await uow.jobs.save(job)
            await uow.commit()
        return JobResult(job=job, item=current)

    async def _document_of(self, item: SourceItem) -> Document | None:
        async with self._uow_factory() as uow:
            return await uow.documents.for_item(item.id)

    async def _unchanged_file(self, item: SourceItem) -> DiscoveredFile:
        file = await asyncio.to_thread(self._source.discover, item.relative_path)
        if file is None or file.source_version != item.source_version:
            raise SourceChangedError("the file changed or disappeared after it was scanned")
        return file

    def _check_size(self, parsed: ParsedDocument, chunks: tuple[ChunkDraft, ...]) -> None:
        characters = sum(len(block.content) for block in parsed.blocks)
        if characters > self._limits.max_document_characters:
            raise InvalidDocumentError(
                f"extracted text has {characters} characters; the limit is "
                f"{self._limits.max_document_characters}"
            )
        if not chunks:
            raise InvalidDocumentError("the chunker produced no chunks")
        if len(chunks) > self._limits.max_document_chunks:
            raise InvalidDocumentError(
                f"the document produced {len(chunks)} chunks; the limit is "
                f"{self._limits.max_document_chunks}"
            )


async def _leased(
    uow: UnitOfWork, job_id: JobId, worker_id: str, now: datetime
) -> tuple[IngestionJob, SourceItem]:
    job = await uow.jobs.get(job_id)
    if job is None:
        raise LeaseLostError("the job no longer exists")
    job.assert_lease(worker_id, now)
    item = await uow.items.get(job.source_item_id)
    if item is None:
        raise LeaseLostError("the source item no longer exists")
    return job, item
