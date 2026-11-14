"""retrieval 依賴的儲存與模型 port，以及它們之間傳遞的資料。

Qdrant 與 PostgreSQL 只回傳 chunk ID 與分數；權限複驗與組裝證據一律以 `ChunkCatalog` 讀出的
PostgreSQL 最新狀態為準。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from helpdesk.domain.access import Principal
from helpdesk.domain.documents import Chunk, ChunkId, Document, IndexingStatus
from helpdesk.domain.errors import DomainValidationError


class QueryTooLongError(DomainValidationError):
    """問題超過 embedding model 的 token 上限。"""


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk_id: ChunkId
    score: float


@dataclass(frozen=True, slots=True)
class StoredChunk:
    chunk: Chunk
    document: Document

    @property
    def is_current(self) -> bool:
        """兩種權限模式都只採用文件目前版本、已寫進索引的 chunk。"""

        return (
            self.chunk.status is IndexingStatus.INDEXED
            and self.document.current_version_id == self.chunk.version_id
        )


@dataclass(frozen=True, slots=True)
class CurrentChunks:
    """向量檢索只限目前版本，不看 ACL 與文件狀態。"""


@dataclass(frozen=True, slots=True)
class ReadableBy:
    """向量檢索只限目前版本、已發布，而且公開或允許這個身分、也沒有拒絕它的 chunk。"""

    principal: Principal


VectorScope = CurrentChunks | ReadableBy


@dataclass(frozen=True, slots=True)
class KeywordTerm:
    text: str
    weight: int


@dataclass(frozen=True, slots=True)
class KeywordQuery:
    """chunk 的分數是它包含的詞的權重總和，低於 `min_score` 不算命中。"""

    terms: tuple[KeywordTerm, ...]
    min_score: int


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]:
        """同步執行，use case 會放到 thread。超過 token 上限時丟 `QueryTooLongError`。"""
        ...


class VectorSearch(Protocol):
    async def search(
        self, vector: Sequence[float], scope: VectorScope, *, limit: int, offset: int
    ) -> list[ScoredChunk]: ...


class KeywordSearch(Protocol):
    async def search(
        self, query: KeywordQuery, *, limit: int, offset: int
    ) -> list[ScoredChunk]: ...


class ChunkCatalog(Protocol):
    async def load(self, chunk_ids: Sequence[ChunkId]) -> list[StoredChunk]:
        """回傳找得到的 chunk 與所屬文件（含存取規則），順序不保證。"""
        ...
