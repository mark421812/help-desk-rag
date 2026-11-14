"""ACL 開關在 ingestion 的兩種實作，由 composition root 依設定選一個注入。

`OptionalMetadata`（ACL 關閉）：metadata 檔案選填。沒寫的標題用檔名，沒寫的狀態視為已發布、
可外送，其他欄位用 domain 預設。`RequiredMetadata`（ACL 開啟）：必填欄位缺一個就
拒絕，沒寫的治理欄位用 domain 的保守預設。兩者都完整保存存取規則。
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Final, TypeVar

from helpdesk.application.ingestion.ports import DeclaredMetadata
from helpdesk.domain.documents import DocumentMetadata, DocumentStatus
from helpdesk.domain.errors import DomainValidationError
from helpdesk.domain.ingestion import InvalidMetadataError, MetadataMissingError

REQUIRED_FIELDS: Final = ("title", "document_type", "owner_department", "document_owner")

_T = TypeVar("_T")
_DOMAIN_DEFAULTS: Final = DocumentMetadata(title="domain defaults")


class OptionalMetadata:
    name: Final = "optional-metadata-v1"

    def resolve(self, declared: DeclaredMetadata, relative_path: str) -> DocumentMetadata:
        return _metadata(
            declared,
            title=declared.title or PurePosixPath(relative_path).stem,
            status=_or(declared.status, DocumentStatus.PUBLISHED),
            external_llm_allowed=_or(declared.external_llm_allowed, True),
        )


class RequiredMetadata:
    name: Final = "required-metadata-v1"

    def resolve(self, declared: DeclaredMetadata, relative_path: str) -> DocumentMetadata:
        missing = tuple(name for name in REQUIRED_FIELDS if getattr(declared, name) is None)
        title = declared.title
        if missing or title is None:
            raise MetadataMissingError(missing)
        return _metadata(
            declared,
            title=title,
            status=_or(declared.status, _DOMAIN_DEFAULTS.status),
            external_llm_allowed=_or(
                declared.external_llm_allowed, _DOMAIN_DEFAULTS.external_llm_allowed
            ),
        )


def _or(value: _T | None, default: _T) -> _T:
    return default if value is None else value


def _metadata(
    declared: DeclaredMetadata,
    *,
    title: str,
    status: DocumentStatus,
    external_llm_allowed: bool,
) -> DocumentMetadata:
    try:
        return DocumentMetadata(
            title=title,
            document_type=_or(declared.document_type, _DOMAIN_DEFAULTS.document_type),
            language=_or(declared.language, _DOMAIN_DEFAULTS.language),
            status=status,
            confidentiality=_or(declared.confidentiality, _DOMAIN_DEFAULTS.confidentiality),
            external_llm_allowed=external_llm_allowed,
            effective_at=declared.effective_at,
            expires_at=declared.expires_at,
            access_rules=declared.access_rules,
        )
    except DomainValidationError as exc:
        raise InvalidMetadataError(str(exc)) from exc
