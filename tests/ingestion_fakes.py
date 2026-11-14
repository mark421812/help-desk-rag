"""ingestion 單元測試用的替身：記憶體版 unit of work、向量索引、embedding 與時鐘。

檔案系統、parser、切塊與 metadata YAML 都用正式的 adapter，只換掉資料庫、Qdrant 與模型。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Self

from helpdesk.application.ingestion.metadata import OptionalMetadata, RequiredMetadata
from helpdesk.application.ingestion.ports import MetadataPolicy, UnitOfWork
from helpdesk.application.ingestion.processing import ClaimNextJob, ProcessingLimits, ProcessJob
from helpdesk.application.ingestion.scanning import (
    DeletionRules,
    Discovery,
    IngestDocument,
    ScanDocuments,
)
from helpdesk.domain.documents import (
    Chunk,
    ChunkId,
    Document,
    DocumentId,
    DocumentVersion,
    IndexingStatus,
    SourceItemId,
    VersionId,
)
from helpdesk.domain.ingestion import (
    IngestionJob,
    IngestionJobStatus,
    JobId,
    SourceItem,
    SourceItemStatus,
)
from helpdesk.infrastructure.chunking.structure_aware import ChunkingConfig, StructureAwareChunker
from helpdesk.infrastructure.metadata.yaml_sidecars import YamlSidecarMetadata
from helpdesk.infrastructure.parsers.registry import default_parsers
from helpdesk.infrastructure.sources.local_folder import LocalFolderSource

WORKER = "worker-1"


class WordCounter:
    def count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 13, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


class FakeEmbedder:
    """以內容雜湊產生固定的單位向量，同樣的文字永遠得到同樣的向量。"""

    model_id = "fake-e5"
    revision = "rev-1"

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension
        self.embedded: list[str] = []

    def dimension(self) -> int:
        return self._dimension

    def embed(self, passages: Sequence[str]) -> list[list[float]]:
        self.embedded.extend(passages)
        return [self._vector(passage) for passage in passages]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        raw = [digest[index] + 1.0 for index in range(self._dimension)]
        norm = math.sqrt(sum(value * value for value in raw))
        return [value / norm for value in raw]


@dataclass
class Point:
    document_id: str
    version_id: str
    is_current: bool
    title: str


@dataclass
class FakeIndex:
    """記住每個 point 的版本與是否為目前版本，並記下每次寫入時 PostgreSQL 的狀態。"""

    points: dict[str, Point] = field(default_factory=dict)
    dimensions: list[int] = field(default_factory=list)
    on_upsert: Callable[[Document], None] | None = None
    fail_publish: int = 0

    async def ensure_collection(self, dimension: int) -> None:
        self.dimensions.append(dimension)

    async def upsert(
        self, document: Document, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        assert len(chunks) == len(vectors)
        for chunk in chunks:
            self.points[str(chunk.id)] = Point(
                document_id=str(document.id),
                version_id=str(chunk.version_id),
                is_current=False,
                title=document.metadata.title,
            )
        if self.on_upsert is not None:
            self.on_upsert(document)

    async def publish(self, document: Document) -> None:
        if self.fail_publish:
            self.fail_publish -= 1
            raise ConnectionError("qdrant is unavailable")
        current = str(document.current_version_id)
        for point_id, point in list(self.points.items()):
            if point.document_id != str(document.id):
                continue
            if point.version_id == current:
                point.is_current = True
                point.title = document.metadata.title
            else:
                del self.points[point_id]

    async def delete_document(self, document_id: DocumentId) -> None:
        kept = {key: p for key, p in self.points.items() if p.document_id != str(document_id)}
        self.points = kept

    def current_points(self) -> list[Point]:
        return [point for point in self.points.values() if point.is_current]


@dataclass
class Store:
    items: dict[SourceItemId, SourceItem] = field(default_factory=dict)
    jobs: dict[JobId, IngestionJob] = field(default_factory=dict)
    documents: dict[DocumentId, Document] = field(default_factory=dict)
    versions: dict[VersionId, DocumentVersion] = field(default_factory=dict)
    chunks: dict[ChunkId, Chunk] = field(default_factory=dict)
    last_full_scan: datetime | None = None

    def copy(self) -> Store:
        return Store(
            dict(self.items),
            dict(self.jobs),
            dict(self.documents),
            dict(self.versions),
            dict(self.chunks),
            self.last_full_scan,
        )

    def item(self, relative_path: str) -> SourceItem:
        return next(item for item in self.items.values() if item.relative_path == relative_path)

    def document_of(self, relative_path: str) -> Document | None:
        item = self.item(relative_path)
        return next((doc for doc in self.documents.values() if doc.source_item_id == item.id), None)

    def chunks_of(self, version_id: VersionId) -> list[Chunk]:
        return sorted(
            (chunk for chunk in self.chunks.values() if chunk.version_id == version_id),
            key=lambda chunk: chunk.chunk_index,
        )


class _Items:
    def __init__(self, data: Store) -> None:
        self._data = data

    async def get(self, item_id: SourceItemId) -> SourceItem | None:
        return self._data.items.get(item_id)

    async def by_path(self, relative_path: str) -> SourceItem | None:
        return next(
            (item for item in self._data.items.values() if item.relative_path == relative_path),
            None,
        )

    async def all(self) -> list[SourceItem]:
        return sorted(self._data.items.values(), key=lambda item: item.relative_path)

    async def save(self, item: SourceItem) -> None:
        self._data.items[item.id] = item

    async def page(
        self, *, status: SourceItemStatus | None, offset: int, limit: int
    ) -> list[SourceItem]:
        items = [item for item in await self.all() if status is None or item.status is status]
        return items[offset : offset + limit]


class _FullScans:
    def __init__(self, data: Store) -> None:
        self._data = data

    async def record_completed(self, at: datetime) -> None:
        self._data.last_full_scan = at

    async def last_completed_at(self) -> datetime | None:
        return self._data.last_full_scan


class _Jobs:
    def __init__(self, data: Store) -> None:
        self._data = data

    async def get(self, job_id: JobId) -> IngestionJob | None:
        return self._data.jobs.get(job_id)

    async def add_if_absent(self, job: IngestionJob) -> bool:
        if any(other.idempotency_key == job.idempotency_key for other in self._data.jobs.values()):
            return False
        self._data.jobs[job.id] = job
        return True

    async def save(self, job: IngestionJob) -> None:
        self._data.jobs[job.id] = job

    async def claim_next(
        self, worker_id: str, now: datetime, lease_seconds: float
    ) -> IngestionJob | None:
        running = {
            job.source_item_id
            for job in self._data.jobs.values()
            if job.status is IngestionJobStatus.PROCESSING and not job.is_claimable(now)
        }
        for job in sorted(self._data.jobs.values(), key=lambda job: job.available_at):
            if job.is_claimable(now) and job.source_item_id not in running:
                claimed = job.claim(worker_id, now, timedelta(seconds=lease_seconds))
                self._data.jobs[job.id] = claimed
                return claimed
        return None

    async def latest_by_item(
        self, item_ids: Sequence[SourceItemId]
    ) -> Mapping[SourceItemId, IngestionJob]:
        latest: dict[SourceItemId, IngestionJob] = {}
        for job in self._data.jobs.values():
            if job.source_item_id in item_ids:
                latest[job.source_item_id] = job
        return latest


class _Documents:
    def __init__(self, data: Store) -> None:
        self._data = data

    async def for_item(self, item_id: SourceItemId) -> Document | None:
        return next(
            (doc for doc in self._data.documents.values() if doc.source_item_id == item_id), None
        )

    async def save_metadata(self, document: Document) -> None:
        existing = self._data.documents.get(document.id)
        current = existing.current_version_id if existing else None
        self._data.documents[document.id] = replace(document, current_version_id=current)

    async def version(
        self, document_id: DocumentId, content_hash: str, model_id: str, revision: str
    ) -> DocumentVersion | None:
        return next(
            (
                version
                for version in self._data.versions.values()
                if (
                    version.document_id,
                    version.content_hash,
                    version.embedding_model,
                    version.embedding_model_revision,
                )
                == (document_id, content_hash, model_id, revision)
            ),
            None,
        )

    async def add_version(self, version: DocumentVersion, chunks: Sequence[Chunk]) -> None:
        assert version.id not in self._data.versions
        self._data.versions[version.id] = version
        for chunk in chunks:
            self._data.chunks[chunk.id] = chunk

    async def chunks(self, version_id: VersionId) -> list[Chunk]:
        return self._data.chunks_of(version_id)

    async def activate(self, document_id: DocumentId, version_id: VersionId) -> None:
        for chunk in self._data.chunks_of(version_id):
            self._data.chunks[chunk.id] = replace(
                chunk, status=IndexingStatus.INDEXED, qdrant_point_id=chunk.id
            )
        self._data.versions[version_id] = replace(
            self._data.versions[version_id], status=IndexingStatus.INDEXED
        )
        self._data.documents[document_id] = replace(
            self._data.documents[document_id], current_version_id=version_id
        )
        for other in [v for v in self._data.versions.values() if v.document_id == document_id]:
            if other.id != version_id:
                self._drop_version(other.id)

    async def delete(self, document_id: DocumentId) -> None:
        self._data.documents.pop(document_id, None)
        for version in [v for v in self._data.versions.values() if v.document_id == document_id]:
            self._drop_version(version.id)

    def _drop_version(self, version_id: VersionId) -> None:
        del self._data.versions[version_id]
        for chunk in self._data.chunks_of(version_id):
            del self._data.chunks[chunk.id]


class MemoryUnitOfWork:
    """每個 unit of work 在複本上工作，commit 才寫回；沒 commit 就離開等於 rollback。"""

    def __init__(self, store: Store) -> None:
        self._store = store

    async def __aenter__(self) -> Self:
        self._working = self._store.copy()
        self._items = _Items(self._working)
        self._jobs = _Jobs(self._working)
        self._documents = _Documents(self._working)
        self._full_scans = _FullScans(self._working)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    @property
    def items(self) -> _Items:
        return self._items

    @property
    def jobs(self) -> _Jobs:
        return self._jobs

    @property
    def documents(self) -> _Documents:
        return self._documents

    @property
    def full_scans(self) -> _FullScans:
        return self._full_scans

    async def commit(self) -> None:
        for name in ("items", "jobs", "documents", "versions", "chunks", "last_full_scan"):
            setattr(self._store, name, getattr(self._working, name))
        self._working = self._store.copy()
        self._items = _Items(self._working)
        self._jobs = _Jobs(self._working)
        self._documents = _Documents(self._working)
        self._full_scans = _FullScans(self._working)


class NoScanLock:
    @asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        yield


@dataclass
class Pipeline:
    """一套接好的 ingestion：正式的檔案、parser、切塊與 YAML adapter，加上替身。"""

    root: Path
    access_control: bool = False
    store: Store = field(default_factory=Store)
    clock: FakeClock = field(default_factory=FakeClock)
    embedder: FakeEmbedder = field(default_factory=FakeEmbedder)
    index: FakeIndex = field(default_factory=FakeIndex)
    max_file_size_bytes: int = 10_000_000

    @property
    def policy(self) -> MetadataPolicy:
        return RequiredMetadata() if self.access_control else OptionalMetadata()

    def unit_of_work(self) -> UnitOfWork:
        return MemoryUnitOfWork(self.store)

    def _discovery(self) -> Discovery:
        parsers = default_parsers()
        return Discovery(
            source=LocalFolderSource(self.root, parsers.supported_extensions),
            metadata_source=YamlSidecarMetadata(self.root),
            policy_name=self.policy.name,
            max_attempts=3,
        )

    def scanner(self) -> ScanDocuments:
        return ScanDocuments(
            self.unit_of_work, self._discovery(), NoScanLock(), DeletionRules(), self.clock
        )

    def ingester(self) -> IngestDocument:
        return IngestDocument(self.unit_of_work, self._discovery(), NoScanLock(), self.clock)

    def processor(self) -> ProcessJob:
        parsers = default_parsers()
        return ProcessJob(
            self.unit_of_work,
            source=LocalFolderSource(self.root, parsers.supported_extensions),
            parser=parsers,
            chunker=StructureAwareChunker(
                WordCounter(), ChunkingConfig(target_tokens=12, max_tokens=16, overlap_tokens=2)
            ),
            embedder=self.embedder,
            metadata_source=YamlSidecarMetadata(self.root),
            metadata_policy=self.policy,
            index=self.index,
            limits=ProcessingLimits(
                max_file_size_bytes=self.max_file_size_bytes,
                retry_base_delay=timedelta(seconds=10),
            ),
            clock=self.clock,
        )

    async def drain(self) -> int:
        """處理所有現在可以執行的 job，回傳處理了幾個。"""

        claim = ClaimNextJob(self.unit_of_work, timedelta(minutes=10), self.clock)
        processor = self.processor()
        handled = 0
        while (job := await claim.run(WORKER)) is not None:
            await processor.run(job.id, WORKER)
            handled += 1
        return handled
