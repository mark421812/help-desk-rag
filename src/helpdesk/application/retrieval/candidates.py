"""粗檢索：E5 查詢向量搜 Qdrant、PostgreSQL 關鍵字搜尋，各自通過權限複驗後以 RRF 合併。

兩路各自分頁抓候選、讀回 PostgreSQL 的
最新狀態並複驗，湊滿 `limit * 3` 個或抓到上限為止；被拒絕的 chunk 因此不會佔掉 RRF 的名次。
關鍵字那一路失敗時只用向量結果，`keyword_failure` 記下例外型別，不記任何內容。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from helpdesk.application.retrieval.keywords import keyword_query
from helpdesk.application.retrieval.ports import (
    ChunkCatalog,
    KeywordSearch,
    QueryEmbedder,
    ScoredChunk,
    StoredChunk,
    VectorScope,
    VectorSearch,
)
from helpdesk.domain.documents import ChunkId
from helpdesk.domain.retrieval import Candidate

RRF_K: Final = 60
MAX_SEARCH_CANDIDATES: Final = 500
_POOL_FACTOR: Final = 3

Fetch = Callable[[int, int], Awaitable[list[ScoredChunk]]]
Ranking = list[tuple[StoredChunk, float]]


@dataclass(frozen=True, slots=True)
class CandidatePool:
    """依 RRF 分數排好的候選，以及每個候選讀回的 chunk 與文件。"""

    candidates: tuple[Candidate, ...]
    chunks: Mapping[ChunkId, StoredChunk]
    keyword_failure: str | None = None


class FindCandidates:
    def __init__(
        self,
        *,
        embedder: QueryEmbedder,
        vectors: VectorSearch,
        keywords: KeywordSearch,
        catalog: ChunkCatalog,
        keyword_min_score: int,
    ) -> None:
        self._embedder = embedder
        self._vectors = vectors
        self._keywords = keywords
        self._catalog = catalog
        self._keyword_min_score = keyword_min_score

    async def run(
        self,
        question: str,
        *,
        scope: VectorScope,
        admit: Callable[[StoredChunk], bool],
        limit: int,
    ) -> CandidatePool:
        pool = min(limit * _POOL_FACTOR, MAX_SEARCH_CANDIDATES)
        query = keyword_query(question, self._keyword_min_score)

        async def vector_side() -> Ranking:
            vector = await asyncio.to_thread(self._embedder.embed_query, question)

            async def fetch(size: int, offset: int) -> list[ScoredChunk]:
                return await self._vectors.search(vector, scope, limit=size, offset=offset)

            return await self._collect(fetch, admit, pool)

        async def keyword_side() -> Ranking:
            if not query.terms:
                return []

            async def fetch(size: int, offset: int) -> list[ScoredChunk]:
                return await self._keywords.search(query, limit=size, offset=offset)

            return await self._collect(fetch, admit, pool)

        vector_ranking, keyword_ranking = await asyncio.gather(
            vector_side(), keyword_side(), return_exceptions=True
        )
        if isinstance(vector_ranking, BaseException):
            raise vector_ranking
        keyword_failure: str | None = None
        if isinstance(keyword_ranking, Exception):
            keyword_failure = type(keyword_ranking).__name__
            rankings = [vector_ranking]
        elif isinstance(keyword_ranking, BaseException):
            raise keyword_ranking
        else:
            rankings = [vector_ranking, keyword_ranking]
        return CandidatePool(
            candidates=fuse(rankings, limit),
            chunks={stored.chunk.id: stored for ranking in rankings for stored, _ in ranking},
            keyword_failure=keyword_failure,
        )

    async def _collect(
        self, fetch: Fetch, admit: Callable[[StoredChunk], bool], target: int
    ) -> Ranking:
        offset = 0
        page_size = min(target * _POOL_FACTOR, MAX_SEARCH_CANDIDATES)
        seen: set[ChunkId] = set()
        kept: Ranking = []
        while offset < MAX_SEARCH_CANDIDATES and len(kept) < target:
            size = min(page_size, MAX_SEARCH_CANDIDATES - offset)
            hits = await fetch(size, offset)
            if not hits:
                break
            offset += len(hits)
            fresh = [hit for hit in hits if hit.chunk_id not in seen]
            seen.update(hit.chunk_id for hit in fresh)
            loaded = {
                stored.chunk.id: stored
                for stored in await self._catalog.load([hit.chunk_id for hit in fresh])
            }
            for hit in fresh:
                stored = loaded.get(hit.chunk_id)
                if stored is not None and stored.is_current and admit(stored):
                    kept.append((stored, hit.score))
            if len(hits) < size:
                break
            page_size = min(page_size * 2, MAX_SEARCH_CANDIDATES - offset)
        return kept[:target]


def fuse(rankings: Sequence[Ranking], limit: int) -> tuple[Candidate, ...]:
    """reciprocal rank fusion，分數除以所有排名都第 1 名時的分數，落在 (0, 1]。

    第一個排名是向量、第二個是關鍵字。同分時保留第一次出現的順序，也就是向量優先。
    """

    fused: dict[ChunkId, float] = {}
    scores: list[dict[ChunkId, float]] = []
    for ranking in rankings:
        side: dict[ChunkId, float] = {}
        for rank, (stored, score) in enumerate(ranking, start=1):
            chunk_id = stored.chunk.id
            side[chunk_id] = score
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        scores.append(side)
    best_possible = len(rankings) / (RRF_K + 1)
    vector_scores = scores[0] if scores else {}
    keyword_scores = scores[1] if len(scores) > 1 else {}
    ordered = sorted(fused, key=lambda chunk_id: -fused[chunk_id])
    return tuple(
        Candidate(
            chunk_id=chunk_id,
            fused_score=fused[chunk_id] / best_possible,
            vector_score=vector_scores.get(chunk_id),
            keyword_score=keyword_scores.get(chunk_id),
        )
        for chunk_id in ordered[:limit]
    )
