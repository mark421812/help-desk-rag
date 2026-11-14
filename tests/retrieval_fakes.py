"""retrieval 單元測試用的記憶體 adapter 與資料建構。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from uuid import uuid4

from helpdesk.application.outbound_scan import ScanVerdict
from helpdesk.application.retrieval.ports import (
    KeywordQuery,
    ScoredChunk,
    StoredChunk,
    VectorScope,
)
from helpdesk.domain.access import AccessRule
from helpdesk.domain.documents import (
    Chunk,
    ChunkId,
    Document,
    DocumentId,
    DocumentMetadata,
    DocumentStatus,
    IndexingStatus,
    SourceItemId,
    VersionId,
)

TODAY = date(2026, 11, 13)


def document(
    *,
    title: str = "115 碩專簡章",
    status: DocumentStatus = DocumentStatus.PUBLISHED,
    external_llm_allowed: bool = True,
    effective_at: date | None = None,
    expires_at: date | None = None,
    rules: tuple[AccessRule, ...] = (),
) -> Document:
    return Document(
        id=DocumentId(uuid4()),
        source_item_id=SourceItemId(uuid4()),
        metadata=DocumentMetadata(
            title=title,
            document_type="policy",
            status=status,
            external_llm_allowed=external_llm_allowed,
            effective_at=effective_at,
            expires_at=expires_at,
            access_rules=rules,
        ),
        current_version_id=VersionId(uuid4()),
    )


def stored(
    owner: Document | None = None,
    *,
    content: str | None = None,
    tokens: int = 10,
    page: int | None = 1,
    current: bool = True,
    indexed: bool = True,
) -> StoredChunk:
    owner = owner or document()
    assert owner.current_version_id is not None
    chunk_id = ChunkId(uuid4())
    return StoredChunk(
        chunk=Chunk(
            id=chunk_id,
            document_id=owner.id,
            version_id=owner.current_version_id if current else VersionId(uuid4()),
            chunk_index=0,
            content=content or f"內容 {chunk_id}",
            token_count=tokens,
            section_path="報名",
            page_number=page,
            status=IndexingStatus.INDEXED if indexed else IndexingStatus.PENDING,
            qdrant_point_id=chunk_id if indexed else None,
        ),
        document=owner,
    )


class FixedEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]


@dataclass
class Ranked:
    """依給定順序回傳命中，模擬 Qdrant 或 PostgreSQL 的排序與分頁。"""

    chunks: list[StoredChunk] = field(default_factory=list)
    error: Exception | None = None
    calls: list[tuple[int, int]] = field(default_factory=list)

    def page(self, limit: int, offset: int) -> list[ScoredChunk]:
        self.calls.append((limit, offset))
        if self.error is not None:
            raise self.error
        window = self.chunks[offset : offset + limit]
        return [
            ScoredChunk(stored.chunk.id, float(len(self.chunks) - offset - index))
            for index, stored in enumerate(window)
        ]


class FakeVectors(Ranked):
    def __init__(self, chunks: Sequence[StoredChunk] = ()) -> None:
        super().__init__(list(chunks))
        self.scopes: list[VectorScope] = []

    async def search(
        self, vector: Sequence[float], scope: VectorScope, *, limit: int, offset: int
    ) -> list[ScoredChunk]:
        self.scopes.append(scope)
        return self.page(limit, offset)


class FakeKeywords(Ranked):
    def __init__(self, chunks: Sequence[StoredChunk] = (), error: Exception | None = None) -> None:
        super().__init__(list(chunks), error)
        self.queries: list[KeywordQuery] = []

    async def search(self, query: KeywordQuery, *, limit: int, offset: int) -> list[ScoredChunk]:
        self.queries.append(query)
        return self.page(limit, offset)


class FakeCatalog:
    def __init__(self, chunks: Sequence[StoredChunk]) -> None:
        self._chunks = {stored.chunk.id: stored for stored in chunks}

    async def load(self, chunk_ids: Sequence[ChunkId]) -> list[StoredChunk]:
        return [self._chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self._chunks]


class RecordingScan:
    def __init__(self, *blocked_rules: str) -> None:
        self.scanned: list[list[str]] = []
        self._blocked = blocked_rules

    def scan(self, texts: Sequence[str]) -> ScanVerdict:
        self.scanned.append(list(texts))
        return ScanVerdict(policy_version="test", rule_ids=self._blocked)
