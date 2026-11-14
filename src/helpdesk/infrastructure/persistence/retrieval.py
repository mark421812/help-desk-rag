"""retrieval 的 PostgreSQL adapter：關鍵字檢索與讀回 chunk 的最新狀態。

關鍵字檢索不靠額外索引。chunk 全文只存在 PostgreSQL，直接用 ILIKE 比對每個
詞，分數是命中詞權重的總和，不需要 extension 或額外的資料表。SQL 只篩目前版本、已索引的
chunk；ACL、狀態與日期由 `AccessPolicy.admits` 複驗，和向量那一路相同。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from sqlalchemy import ColumnElement, Integer, case, literal, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from helpdesk.application.retrieval.ports import KeywordQuery, ScoredChunk, StoredChunk
from helpdesk.domain.documents import ChunkId, IndexingStatus
from helpdesk.infrastructure.persistence.ingestion import chunk_from_row, document_from_row
from helpdesk.infrastructure.persistence.models import (
    ChunkModel,
    DocumentAccessRuleModel,
    DocumentModel,
)


class PostgresKeywordSearch:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def search(self, query: KeywordQuery, *, limit: int, offset: int) -> list[ScoredChunk]:
        if not query.terms:
            return []
        # 權重以帶型別的 literal 綁定，asyncpg 才不會把 CASE 的兩個分支推成 text 而無法相加。
        term_scores: list[ColumnElement[int]] = [
            case(
                (
                    ChunkModel.content.ilike(f"%{_escape_like(term.text)}%", escape="\\"),
                    literal(term.weight, Integer),
                ),
                else_=literal(0, Integer),
            )
            for term in query.terms
        ]
        score = term_scores[0]
        for term_score in term_scores[1:]:
            score = score + term_score
        statement = (
            select(ChunkModel.id, score.label("score"))
            .join(DocumentModel, DocumentModel.id == ChunkModel.document_id)
            .where(
                DocumentModel.current_version_id == ChunkModel.version_id,
                ChunkModel.indexing_status == IndexingStatus.INDEXED.value,
                score >= query.min_score,
            )
            # 同分時依文件與段落順序，而不是每次重建索引都會變的 chunk UUID，排名才可重現。
            .order_by(score.desc(), ChunkModel.document_id, ChunkModel.chunk_index)
            .limit(limit)
            .offset(offset)
        )
        async with self._sessions() as session:
            rows = (await session.execute(statement)).all()
        return [ScoredChunk(chunk_id=ChunkId(row[0]), score=float(row[1])) for row in rows]


class SqlChunkCatalog:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def load(self, chunk_ids: Sequence[ChunkId]) -> list[StoredChunk]:
        if not chunk_ids:
            return []
        async with self._sessions() as session:
            pairs = (
                await session.execute(
                    select(ChunkModel, DocumentModel)
                    .join(DocumentModel, DocumentModel.id == ChunkModel.document_id)
                    .where(ChunkModel.id.in_(list(chunk_ids)))
                )
            ).all()
            document_ids = {document.id for _, document in pairs}
            rules: defaultdict[object, list[DocumentAccessRuleModel]] = defaultdict(list)
            if document_ids:
                for rule in await session.scalars(
                    select(DocumentAccessRuleModel).where(
                        DocumentAccessRuleModel.document_id.in_(document_ids)
                    )
                ):
                    rules[rule.document_id].append(rule)
        documents = {
            document.id: document_from_row(document, rules[document.id]) for _, document in pairs
        }
        return [
            StoredChunk(chunk=chunk_from_row(chunk), document=documents[document.id])
            for chunk, document in pairs
        ]

    async def is_ready(self) -> bool:
        try:
            async with self._sessions() as session:
                await session.execute(select(literal(1)))
        except Exception:
            return False
        return True


def _escape_like(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
