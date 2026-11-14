from __future__ import annotations

import hashlib
from datetime import date
from uuid import uuid4

import pytest

from helpdesk.domain.access import AccessRule, PermissionEffect, PrincipalType
from helpdesk.domain.documents import (
    Chunk,
    ChunkId,
    Confidentiality,
    DocumentId,
    DocumentMetadata,
    DocumentStatus,
    DocumentVersion,
    IndexingStatus,
    VersionId,
)
from helpdesk.domain.errors import DomainValidationError


def test_a_title_alone_is_enough_and_governance_defaults_stay_closed() -> None:
    metadata = DocumentMetadata(title="BookRAG_繁體中文完整翻譯")

    assert metadata.document_type == "general"
    assert metadata.language == "zh-TW"
    assert metadata.status is DocumentStatus.DRAFT
    assert metadata.confidentiality is Confidentiality.INTERNAL
    assert metadata.external_llm_allowed is False
    assert metadata.access_rules == ()
    assert metadata.effective_at is None
    assert metadata.expires_at is None


@pytest.mark.parametrize("level", [Confidentiality.CONFIDENTIAL, Confidentiality.RESTRICTED])
def test_confidential_documents_need_an_allow_rule(level: Confidentiality) -> None:
    deny_only = (AccessRule(PrincipalType.USER, "alice", PermissionEffect.DENY),)
    with pytest.raises(DomainValidationError):
        DocumentMetadata(title="薪資表", confidentiality=level, access_rules=deny_only)

    allowed = DocumentMetadata(
        title="薪資表",
        confidentiality=level,
        access_rules=(AccessRule(PrincipalType.GROUP, "hr"),),
    )
    assert allowed.access_rules[0].effect is PermissionEffect.ALLOW


def test_duplicate_access_rules_are_rejected() -> None:
    rule = AccessRule(PrincipalType.GROUP, "hr")
    with pytest.raises(DomainValidationError):
        DocumentMetadata(title="薪資表", access_rules=(rule, rule))


def test_a_document_cannot_expire_before_it_takes_effect() -> None:
    with pytest.raises(DomainValidationError):
        DocumentMetadata(
            title="簡章", effective_at=date(2026, 11, 1), expires_at=date(2026, 10, 31)
        )


def chunk(**fields: object) -> Chunk:
    values: dict[str, object] = {
        "id": ChunkId(uuid4()),
        "document_id": DocumentId(uuid4()),
        "version_id": VersionId(uuid4()),
        "chunk_index": 0,
        "content": "報名資格：大學畢業並有兩年工作經驗。",
        "token_count": 21,
    }
    return Chunk(**(values | fields))  # type: ignore[arg-type]


def test_chunk_hash_is_derived_from_its_content() -> None:
    content = "報名資格：大學畢業並有兩年工作經驗。"

    assert chunk(content=content).content_hash == hashlib.sha256(content.encode()).hexdigest()


@pytest.mark.parametrize(
    ("status", "point_id", "valid"),
    [
        (IndexingStatus.PENDING, None, True),
        (IndexingStatus.INDEXED, uuid4(), True),
        (IndexingStatus.INDEXED, None, False),
        (IndexingStatus.PENDING, uuid4(), False),
    ],
)
def test_only_indexed_chunks_have_a_qdrant_point(
    status: IndexingStatus, point_id: object, valid: bool
) -> None:
    if valid:
        assert chunk(status=status, qdrant_point_id=point_id).status is status
    else:
        with pytest.raises(DomainValidationError):
            chunk(status=status, qdrant_point_id=point_id)


def test_a_version_needs_a_real_content_hash() -> None:
    with pytest.raises(DomainValidationError):
        DocumentVersion(
            id=VersionId(uuid4()),
            document_id=DocumentId(uuid4()),
            content_hash="mtime:1;size:10",
            mime_type="application/pdf",
            file_size=10,
            parser_name="pypdf",
            parser_version="1",
            embedding_model="intfloat/multilingual-e5-small",
            embedding_model_revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
            embedding_dimension=384,
            chunk_count=1,
        )
