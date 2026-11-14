"""ingestion-worker 的資料形狀與 port。

管線依序是：`DocumentSource` 找到檔案 → `DocumentParser` 轉成 `ParsedDocument` →
`MetadataSource` 讀出 `DeclaredMetadata`，由 `MetadataPolicy` 補成 `DocumentMetadata` →
`Chunker` 切成 `ChunkDraft` → `PassageEmbedder` 產生向量 → `VectorIndex` 與 `UnitOfWork`
寫入 Qdrant 與 PostgreSQL。檔案系統、模型與資料庫的呼叫都是同步或 async 的 adapter，
use case 只看得到這些 Protocol。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

from helpdesk.domain.access import AccessRule
from helpdesk.domain.documents import (
    Chunk,
    Confidentiality,
    Document,
    DocumentId,
    DocumentMetadata,
    DocumentStatus,
    DocumentVersion,
    SourceItemId,
    VersionId,
)
from helpdesk.domain.errors import DomainValidationError
from helpdesk.domain.ingestion import IngestionJob, JobId, SourceItem, SourceItemStatus

Clock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class DiscoveredFile:
    """來源觀察到的一個檔案快照，還沒解析也還沒索引。"""

    relative_path: str
    path: Path
    file_size: int
    modified_at: datetime

    @property
    def source_version(self) -> str:
        return f"mtime:{self.modified_at.isoformat()};size:{self.file_size}"


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    """parser 產生的最小語意區塊，例如一段文字、一個表格。"""

    content: str
    section_path: str | None = None
    page_number: int | None = None
    block_type: str = "paragraph"

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise DomainValidationError("parsed block content must not be blank")
        if self.page_number is not None and self.page_number <= 0:
            raise DomainValidationError("parsed block page_number must be positive")


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    blocks: tuple[ParsedBlock, ...]
    parser_name: str
    parser_version: str
    mime_type: str

    def __post_init__(self) -> None:
        if not self.blocks:
            raise DomainValidationError("a parsed document needs at least one block")


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """切塊結果，還沒有 ID 也還沒有向量。`content` 已含章節路徑前綴。"""

    content: str
    section_path: str | None
    page_number: int | None
    token_count: int


@dataclass(frozen=True, slots=True)
class DeclaredMetadata:
    """metadata 檔案裡明確寫出的欄位；`None` 代表沒寫。

    `owner_department` 與 `document_owner` 只用來檢查 ACL 開啟時的必填規則，不保存。
    """

    title: str | None = None
    document_type: str | None = None
    owner_department: str | None = None
    document_owner: str | None = None
    language: str | None = None
    status: DocumentStatus | None = None
    confidentiality: Confidentiality | None = None
    external_llm_allowed: bool | None = None
    effective_at: date | None = None
    expires_at: date | None = None
    access_rules: tuple[AccessRule, ...] = ()


class DocumentSource(Protocol):
    def scan(self) -> tuple[DiscoveredFile, ...]:
        """回傳來源下所有支援的檔案，依相對路徑排序。"""
        ...

    def discover(self, relative_path: str) -> DiscoveredFile | None:
        """重新觀察一個檔案；檔案不存在或不再支援時回傳 None。"""
        ...


class DocumentParser(Protocol):
    def parse(self, path: Path) -> ParsedDocument: ...


class Chunker(Protocol):
    def chunk(self, document: ParsedDocument) -> tuple[ChunkDraft, ...]: ...


class PassageEmbedder(Protocol):
    """文件段落的 embedding model。方法是同步的，use case 會放到 thread 執行。"""

    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def dimension(self) -> int: ...

    def embed(self, passages: Sequence[str]) -> list[list[float]]: ...


class MetadataSource(Protocol):
    def fingerprint(self, relative_path: str) -> str:
        """生效 metadata 檔案的指紋；內容有問題時也要回傳值，錯誤留給處理階段。"""
        ...

    def declared(self, relative_path: str) -> DeclaredMetadata: ...


class MetadataPolicy(Protocol):
    """ACL 開關在 ingestion 的唯一落點：決定 metadata 是選填還是必填。"""

    @property
    def name(self) -> str: ...

    def resolve(self, declared: DeclaredMetadata, relative_path: str) -> DocumentMetadata: ...


class VectorIndex(Protocol):
    async def ensure_collection(self, dimension: int) -> None: ...

    async def upsert(
        self, document: Document, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        """寫入一個版本的 points，先標成非目前版本。"""
        ...

    async def publish(self, document: Document) -> None:
        """讓 Qdrant 與文件一致：目前版本的 points 帶最新 metadata 並標成目前版本，
        其他版本的 points 刪除。"""
        ...

    async def delete_document(self, document_id: DocumentId) -> None: ...


class ScanLock(Protocol):
    def hold(self) -> AbstractAsyncContextManager[None]:
        """同一時間只有一次掃描；已有掃描在跑時丟出 `ScanInProgressError`。"""
        ...


class SourceItems(Protocol):
    """`get`、`by_path` 與 `all` 會鎖住讀到的資料列直到 commit，掃描與 worker 因此不會
    拿舊資料覆蓋對方剛寫入的狀態。"""

    async def get(self, item_id: SourceItemId) -> SourceItem | None: ...

    async def by_path(self, relative_path: str) -> SourceItem | None: ...

    async def all(self) -> list[SourceItem]: ...

    async def save(self, item: SourceItem) -> None: ...

    async def page(
        self, *, status: SourceItemStatus | None, offset: int, limit: int
    ) -> list[SourceItem]: ...


class FullScans(Protocol):
    """完整掃描的完成紀錄。只有 `ScanDocuments` 寫入，單檔 `ingest` 不算完整掃描。"""

    async def record_completed(self, at: datetime) -> None: ...

    async def last_completed_at(self) -> datetime | None: ...


class IngestionJobs(Protocol):
    async def get(self, job_id: JobId) -> IngestionJob | None: ...

    async def add_if_absent(self, job: IngestionJob) -> bool:
        """同一把 idempotency key 已有 job 時不新增，回傳 False。"""
        ...

    async def save(self, job: IngestionJob) -> None: ...

    async def claim_next(
        self, worker_id: str, now: datetime, lease_seconds: float
    ) -> IngestionJob | None:
        """原子地取下一個可執行的 job；同一個檔案不會同時有兩個 job 在跑。"""
        ...

    async def latest_by_item(
        self, item_ids: Sequence[SourceItemId]
    ) -> Mapping[SourceItemId, IngestionJob]: ...


class Documents(Protocol):
    async def for_item(self, item_id: SourceItemId) -> Document | None: ...

    async def save_metadata(self, document: Document) -> None:
        """新增或更新文件與存取規則，不動 `current_version_id`。"""
        ...

    async def version(
        self, document_id: DocumentId, content_hash: str, model_id: str, revision: str
    ) -> DocumentVersion | None: ...

    async def add_version(self, version: DocumentVersion, chunks: Sequence[Chunk]) -> None: ...

    async def chunks(self, version_id: VersionId) -> list[Chunk]: ...

    async def activate(self, document_id: DocumentId, version_id: VersionId) -> None:
        """把版本與 chunk 標成已索引、切成目前版本，並刪掉其他版本。"""
        ...

    async def delete(self, document_id: DocumentId) -> None: ...


class UnitOfWork(Protocol):
    @property
    def items(self) -> SourceItems: ...

    @property
    def jobs(self) -> IngestionJobs: ...

    @property
    def documents(self) -> Documents: ...

    @property
    def full_scans(self) -> FullScans: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...


UnitOfWorkFactory = Callable[[], UnitOfWork]
