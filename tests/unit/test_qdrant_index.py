"""`ensure_collection` 的冪等行為與 payload 欄位。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient, models

from helpdesk.domain.access import AccessRule, PermissionEffect, PrincipalType
from helpdesk.domain.documents import DocumentMetadata, DocumentStatus
from helpdesk.infrastructure.vector_index.qdrant import (
    PAYLOAD_INDEXES,
    CollectionConflictError,
    QdrantVectorIndex,
    document_payload,
)


class FakeQdrantClient:
    def __init__(self, exists: bool = False, indexed: set[str] | None = None, size: int = 3):
        self.exists = exists
        self.indexed: set[str] = set(indexed or ())
        self.size = size
        self.calls: list[str] = []
        self.fail_index_on: str | None = None
        self.create_collection_error: Exception | None = None

    async def collection_exists(self, name: str) -> bool:
        self.calls.append("collection_exists")
        return self.exists

    async def create_collection(self, **kwargs: Any) -> None:
        self.calls.append("create_collection")
        self.exists = True
        if self.create_collection_error is not None:
            raise self.create_collection_error

    async def get_collection(self, name: str) -> Any:
        self.calls.append("get_collection")
        params = SimpleNamespace(vectors=models.VectorParams(size=self.size, distance="Cosine"))
        return SimpleNamespace(
            config=SimpleNamespace(params=params),
            payload_schema={field: object() for field in self.indexed},
        )

    async def create_payload_index(self, collection_name: str, field_name: str, **kw: Any) -> None:
        self.calls.append(f"index:{field_name}")
        if field_name == self.fail_index_on:
            raise RuntimeError("boom")
        self.indexed.add(field_name)


def index_for(client: FakeQdrantClient) -> QdrantVectorIndex:
    return QdrantVectorIndex(cast(AsyncQdrantClient, client), "helpdesk_chunks_v1")


async def test_indexes_are_created_when_the_collection_already_exists_without_them() -> None:
    client = FakeQdrantClient(exists=True)

    await index_for(client).ensure_collection(3)

    assert client.indexed == set(PAYLOAD_INDEXES)
    assert "create_collection" not in client.calls


async def test_a_collection_created_by_another_process_still_gets_its_indexes() -> None:
    client = FakeQdrantClient(exists=False)
    client.create_collection_error = RuntimeError("already exists")

    await index_for(client).ensure_collection(3)

    assert client.indexed == set(PAYLOAD_INDEXES)


async def test_only_missing_indexes_are_created() -> None:
    client = FakeQdrantClient(exists=True, indexed={"status", "is_current"})

    await index_for(client).ensure_collection(3)

    created = {call.removeprefix("index:") for call in client.calls if call.startswith("index:")}
    assert created == set(PAYLOAD_INDEXES) - {"status", "is_current"}


async def test_a_partial_failure_is_retried_and_resumes() -> None:
    client = FakeQdrantClient(exists=True)
    client.fail_index_on = "language"
    index = index_for(client)
    with pytest.raises(RuntimeError):
        await index.ensure_collection(3)
    assert client.indexed != set(PAYLOAD_INDEXES)

    client.fail_index_on = None
    await index.ensure_collection(3)

    assert client.indexed == set(PAYLOAD_INDEXES)


async def test_ensure_runs_once_per_process_even_under_concurrency() -> None:
    client = FakeQdrantClient(exists=False)
    index = index_for(client)

    await asyncio.gather(*(index.ensure_collection(3) for _ in range(5)))
    await index.ensure_collection(3)

    assert client.calls.count("create_collection") == 1
    assert client.calls.count("get_collection") == 1


async def test_a_dimension_mismatch_is_a_conflict_and_is_not_cached() -> None:
    client = FakeQdrantClient(exists=True, size=8)
    index = index_for(client)

    for _ in range(2):
        with pytest.raises(CollectionConflictError):
            await index.ensure_collection(3)


def test_the_payload_carries_every_field_retrieval_filters_on() -> None:
    metadata = DocumentMetadata(
        title="簡章",
        status=DocumentStatus.PUBLISHED,
        external_llm_allowed=True,
        access_rules=(
            AccessRule(PrincipalType.GROUP, "staff"),
            AccessRule(PrincipalType.USER, "alice"),
            AccessRule(PrincipalType.USER, "bob", PermissionEffect.DENY),
            AccessRule(PrincipalType.GROUP, "interns", PermissionEffect.DENY),
        ),
    )

    payload = document_payload(metadata)

    assert payload == {
        "status": "published",
        "confidentiality": "internal",
        "language": "zh-TW",
        "external_llm_allowed": True,
        "acl_public": False,
        "allowed_user_ids": ["alice"],
        "allowed_group_ids": ["staff"],
        "denied_user_ids": ["bob"],
        "denied_group_ids": ["interns"],
    }
    assert set(payload) | {"document_id", "version_id", "is_current"} == set(PAYLOAD_INDEXES)


def test_a_document_with_only_deny_rules_is_still_public() -> None:
    deny = AccessRule(PrincipalType.USER, str(uuid4()), PermissionEffect.DENY)
    metadata = DocumentMetadata(title="公告", access_rules=(deny,))

    assert document_payload(metadata)["acl_public"] is True
