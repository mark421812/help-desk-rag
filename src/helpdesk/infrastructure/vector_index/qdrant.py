"""Qdrant 向量索引。

Qdrant 只存可重建的向量與檢索前過濾需要的 payload，完整 chunk 與 metadata 以 PostgreSQL
為準。payload 欄位在兩種 ACL 模式下都完整寫入，`AclAccess` 的預先過濾直接用它們。
`QdrantVectorIndex` 給 ingestion-worker 寫入，`QdrantVectorSearch` 給 helpdesk-api 查詢。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Final
from uuid import UUID

from qdrant_client import AsyncQdrantClient, models

from helpdesk.application.retrieval.ports import (
    CurrentChunks,
    ReadableBy,
    ScoredChunk,
    VectorScope,
)
from helpdesk.domain.access import PermissionEffect, PrincipalType
from helpdesk.domain.documents import (
    Chunk,
    ChunkId,
    Document,
    DocumentId,
    DocumentMetadata,
    DocumentStatus,
)
from helpdesk.domain.errors import DomainError

UPSERT_BATCH_SIZE: Final = 256

PAYLOAD_INDEXES: Final[dict[str, models.PayloadSchemaType]] = {
    **dict.fromkeys(
        (
            "document_id",
            "version_id",
            "status",
            "confidentiality",
            "language",
            "allowed_user_ids",
            "allowed_group_ids",
            "denied_user_ids",
            "denied_group_ids",
        ),
        models.PayloadSchemaType.KEYWORD,
    ),
    **dict.fromkeys(
        ("is_current", "acl_public", "external_llm_allowed"), models.PayloadSchemaType.BOOL
    ),
}


class CollectionConflictError(DomainError):
    """既有 collection 的向量維度和目前的 embedding model 不同。"""


def document_payload(metadata: DocumentMetadata) -> dict[str, object]:
    """同一份文件所有 points 共用的過濾欄位。沒有任何 allow 規則的文件視為公開。"""

    def principals(principal_type: PrincipalType, effect: PermissionEffect) -> list[str]:
        return sorted(
            rule.principal_id
            for rule in metadata.access_rules
            if rule.principal_type is principal_type and rule.effect is effect
        )

    return {
        "status": metadata.status.value,
        "confidentiality": metadata.confidentiality.value,
        "language": metadata.language,
        "external_llm_allowed": metadata.external_llm_allowed,
        "acl_public": not any(
            rule.effect is PermissionEffect.ALLOW for rule in metadata.access_rules
        ),
        "allowed_user_ids": principals(PrincipalType.USER, PermissionEffect.ALLOW),
        "allowed_group_ids": principals(PrincipalType.GROUP, PermissionEffect.ALLOW),
        "denied_user_ids": principals(PrincipalType.USER, PermissionEffect.DENY),
        "denied_group_ids": principals(PrincipalType.GROUP, PermissionEffect.DENY),
    }


class QdrantVectorIndex:
    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection
        self._ready_dimension: int | None = None
        self._ensure_lock = asyncio.Lock()

    async def ensure_collection(self, dimension: int) -> None:
        """確認 collection 與所有 payload index 都在。collection 已存在也會補上缺的 index；
        成功後同一個程序不再重做。"""

        if self._ready_dimension == dimension:
            return
        async with self._ensure_lock:
            if self._ready_dimension == dimension:
                return
            await self._ensure(dimension)
            self._ready_dimension = dimension

    async def upsert(
        self, document: Document, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        shared = document_payload(document.metadata)
        points = [
            models.PointStruct(
                id=str(chunk.id),
                vector=list(vector),
                payload={
                    **shared,
                    "chunk_id": str(chunk.id),
                    "document_id": str(chunk.document_id),
                    "version_id": str(chunk.version_id),
                    "is_current": False,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        try:
            for start in range(0, len(points), UPSERT_BATCH_SIZE):
                await self._client.upsert(
                    collection_name=self._collection,
                    points=points[start : start + UPSERT_BATCH_SIZE],
                    wait=True,
                )
        except Exception:
            # collection 可能被外部刪除；下一次寫入前重新確認，而不是永遠相信快取。
            self._ready_dimension = None
            raise

    async def publish(self, document: Document) -> None:
        if document.current_version_id is None:
            raise ValueError("only a document with a current version can be published")
        current = models.FieldCondition(
            key="version_id", match=models.MatchValue(value=str(document.current_version_id))
        )
        await self._client.set_payload(
            collection_name=self._collection,
            payload={**document_payload(document.metadata), "is_current": True},
            points=models.FilterSelector(filter=_document_filter(document.id, must=[current])),
            wait=True,
        )
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=_document_filter(document.id, must_not=[current])
            ),
            wait=True,
        )

    async def delete_document(self, document_id: DocumentId) -> None:
        if not await self._client.collection_exists(self._collection):
            return
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=_document_filter(document_id)),
            wait=True,
        )

    async def close(self) -> None:
        await self._client.close()

    async def _ensure(self, dimension: int) -> None:
        if not await self._client.collection_exists(self._collection):
            try:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=models.VectorParams(
                        size=dimension, distance=models.Distance.COSINE
                    ),
                )
            except Exception:
                # 另一個程序可能剛建好；只有 collection 真的不存在才是錯誤。
                if not await self._client.collection_exists(self._collection):
                    raise
        collection = await self._client.get_collection(self._collection)
        configured = collection.config.params.vectors
        if isinstance(configured, models.VectorParams) and configured.size != dimension:
            raise CollectionConflictError(
                f"collection {self._collection} has dimension {configured.size}, "
                f"expected {dimension}"
            )
        existing = set(collection.payload_schema or {})
        for field, schema in PAYLOAD_INDEXES.items():
            if field not in existing:
                await self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=schema,
                    wait=True,
                )


class QdrantVectorSearch:
    def __init__(self, client: AsyncQdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    async def search(
        self, vector: Sequence[float], scope: VectorScope, *, limit: int, offset: int
    ) -> list[ScoredChunk]:
        result = await self._client.query_points(
            collection_name=self._collection,
            query=list(vector),
            query_filter=vector_filter(scope),
            limit=limit,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        return [
            ScoredChunk(chunk_id=ChunkId(UUID(str(point.id))), score=point.score)
            for point in result.points
        ]

    async def is_ready(self) -> bool:
        try:
            return await self._client.collection_exists(self._collection)
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()


def vector_filter(scope: VectorScope) -> models.Filter:
    """`CurrentChunks` 只限目前版本；`ReadableBy` 再加上已發布、讀得到、沒有被拒絕。"""

    current = models.FieldCondition(key="is_current", match=models.MatchValue(value=True))
    if isinstance(scope, CurrentChunks):
        return models.Filter(must=[current])
    assert isinstance(scope, ReadableBy)
    principal = scope.principal
    readable: list[models.Condition] = [
        models.FieldCondition(key="acl_public", match=models.MatchValue(value=True)),
        models.FieldCondition(
            key="allowed_user_ids", match=models.MatchValue(value=principal.user_id)
        ),
    ]
    denied: list[models.Condition] = [
        models.FieldCondition(
            key="denied_user_ids", match=models.MatchValue(value=principal.user_id)
        ),
    ]
    if principal.group_ids:
        groups = list(principal.group_ids)
        readable.append(
            models.FieldCondition(key="allowed_group_ids", match=models.MatchAny(any=groups))
        )
        denied.append(
            models.FieldCondition(key="denied_group_ids", match=models.MatchAny(any=groups))
        )
    return models.Filter(
        must=[
            current,
            models.FieldCondition(
                key="status", match=models.MatchValue(value=DocumentStatus.PUBLISHED.value)
            ),
            models.Filter(should=readable),
        ],
        must_not=denied,
    )


def _document_filter(
    document_id: DocumentId,
    *,
    must: Sequence[models.Condition] = (),
    must_not: Sequence[models.Condition] = (),
) -> models.Filter:
    return models.Filter(
        must=[
            models.FieldCondition(
                key="document_id", match=models.MatchValue(value=str(document_id))
            ),
            *must,
        ],
        must_not=list(must_not) or None,
    )
