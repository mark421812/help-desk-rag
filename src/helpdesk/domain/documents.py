"""文件、內容版本與 chunk。

一份 `Document` 對應文件根目錄下的一個檔案。每次內容改變產生新的 `DocumentVersion`，
版本底下是可以單獨 embedding 與引用的 `Chunk`。`Document.current_version_id` 指向目前
用來檢索的版本，只有整個版本都寫進索引後才會切換。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Final, NewType
from uuid import UUID

from helpdesk.domain._checks import (
    MAX_LANGUAGE_LENGTH,
    optional_positive,
    optional_text,
    require_positive,
    require_sha256,
    require_text,
    require_unique,
)
from helpdesk.domain.access import AccessRule, PermissionEffect
from helpdesk.domain.errors import DomainValidationError

DocumentId = NewType("DocumentId", UUID)
VersionId = NewType("VersionId", UUID)
ChunkId = NewType("ChunkId", UUID)
SourceItemId = NewType("SourceItemId", UUID)

MAX_TITLE_LENGTH: Final = 1_000
MAX_DOCUMENT_TYPE_LENGTH: Final = 100
MAX_ACCESS_RULES: Final = 1_000


class DocumentStatus(StrEnum):
    DRAFT = "draft"
    REVIEWING = "reviewing"
    PUBLISHED = "published"
    EXPIRED = "expired"
    ARCHIVED = "archived"


class Confidentiality(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class IndexingStatus(StrEnum):
    PENDING = "pending"
    INDEXED = "indexed"


_RULE_REQUIRED: Final = frozenset({Confidentiality.CONFIDENTIAL, Confidentiality.RESTRICTED})


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """文件的描述與治理欄位。

    除了標題，每個欄位都有預設值，所以 ACL 關閉時沒有 metadata 的文件只要給檔名當標題
    就能建立。治理欄位的預設都偏保守（草稿、內部、不可外送、沒有規則）：ACL 關閉時不會
    比對它們，ACL 開啟時它們不會替缺 metadata 的文件放寬任何限制。
    """

    title: str
    document_type: str = "general"
    language: str = "zh-TW"
    status: DocumentStatus = DocumentStatus.DRAFT
    confidentiality: Confidentiality = Confidentiality.INTERNAL
    external_llm_allowed: bool = False
    effective_at: date | None = None
    expires_at: date | None = None
    access_rules: tuple[AccessRule, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.title, "title", max_length=MAX_TITLE_LENGTH)
        require_text(self.document_type, "document_type", max_length=MAX_DOCUMENT_TYPE_LENGTH)
        require_text(self.language, "language", max_length=MAX_LANGUAGE_LENGTH)
        if len(self.access_rules) > MAX_ACCESS_RULES:
            raise DomainValidationError(f"access_rules must not exceed {MAX_ACCESS_RULES}")
        require_unique(self.access_rules, "access_rules")
        if self.effective_at and self.expires_at and self.expires_at < self.effective_at:
            raise DomainValidationError("expires_at must not be before effective_at")
        if self.confidentiality in _RULE_REQUIRED and not any(
            rule.effect is PermissionEffect.ALLOW for rule in self.access_rules
        ):
            raise DomainValidationError(
                "confidential and restricted documents require at least one allow rule"
            )


@dataclass(frozen=True, slots=True)
class Document:
    id: DocumentId
    source_item_id: SourceItemId
    metadata: DocumentMetadata
    current_version_id: VersionId | None = None


@dataclass(frozen=True, slots=True)
class DocumentVersion:
    """一次檔案內容的快照，以及產生它的 parser 與 embedding model。

    `content_hash` 是 parser 名稱、parser 版本與切塊結果的 SHA-256，也就是實際會被索引的
    內容；parser 升級或切塊規則改變都會得到新的版本。內容與 embedding model 都沒變時沿用
    同一個版本，不重做 embedding。
    """

    id: VersionId
    document_id: DocumentId
    content_hash: str
    mime_type: str
    file_size: int
    parser_name: str
    parser_version: str
    embedding_model: str
    embedding_model_revision: str
    embedding_dimension: int
    chunk_count: int
    status: IndexingStatus = IndexingStatus.PENDING

    def __post_init__(self) -> None:
        require_sha256(self.content_hash, "content_hash")
        for name, value in (
            ("mime_type", self.mime_type),
            ("parser_name", self.parser_name),
            ("parser_version", self.parser_version),
            ("embedding_model", self.embedding_model),
            ("embedding_model_revision", self.embedding_model_revision),
        ):
            require_text(value, name, max_length=255)
        if self.file_size < 0:
            raise DomainValidationError("file_size must not be negative")
        require_positive(self.embedding_dimension, "embedding_dimension")
        require_positive(self.chunk_count, "chunk_count")


@dataclass(frozen=True, slots=True)
class Chunk:
    """可以單獨 embedding 與引用的最小內容。

    `qdrant_point_id` 只在 chunk 已寫進 Qdrant 後才有值，兩個欄位一起表達索引狀態。
    """

    id: ChunkId
    document_id: DocumentId
    version_id: VersionId
    chunk_index: int
    content: str
    token_count: int
    section_path: str | None = None
    page_number: int | None = None
    status: IndexingStatus = IndexingStatus.PENDING
    qdrant_point_id: UUID | None = None

    def __post_init__(self) -> None:
        if self.chunk_index < 0:
            raise DomainValidationError("chunk_index must not be negative")
        require_text(self.content, "content")
        require_positive(self.token_count, "token_count")
        optional_text(self.section_path, "section_path")
        optional_positive(self.page_number, "page_number")
        if (self.status is IndexingStatus.INDEXED) != (self.qdrant_point_id is not None):
            raise DomainValidationError("only indexed chunks have a Qdrant point")

    @property
    def content_hash(self) -> str:
        """由內容算出，不另外存放，因此不會和內容對不上。"""

        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()
