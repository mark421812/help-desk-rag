"""讀取 metadata YAML：各層資料夾的 `.ingestion.yaml`，再由單檔的 `<檔名>.metadata.yaml` 覆寫。

越靠近文件的設定優先。這裡只把 YAML 解析成有型別的 `DeclaredMetadata`，哪些欄位必填、
沒寫時用什麼值，交給 ACL 開關選定的 `MetadataPolicy`。`version_number`、
`source_url`、`review_at`、`custom_metadata` 會被接受但不保存。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import date
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Final, TypeVar

import yaml

from helpdesk.application.ingestion.ports import DeclaredMetadata
from helpdesk.domain.access import AccessRule, PermissionEffect, PrincipalType
from helpdesk.domain.documents import Confidentiality, DocumentStatus
from helpdesk.domain.errors import DomainValidationError
from helpdesk.domain.ingestion import InvalidMetadataError

FOLDER_METADATA_FILE: Final = ".ingestion.yaml"
FILE_METADATA_SUFFIX: Final = ".metadata.yaml"
MAX_METADATA_FILE_BYTES: Final = 1024 * 1024

_TEXT_FIELDS: Final = ("title", "document_type", "owner_department", "document_owner", "language")
_IGNORED_FIELDS: Final = frozenset({"version_number", "source_url", "review_at", "custom_metadata"})
_KNOWN_FIELDS: Final = frozenset(
    {
        *_TEXT_FIELDS,
        *_IGNORED_FIELDS,
        "status",
        "confidentiality",
        "external_llm_allowed",
        "effective_at",
        "expires_at",
        "acl",
    }
)
_ACL_FIELDS: Final = frozenset({"principal_type", "principal_id", "effect"})

_Enum = TypeVar("_Enum", bound=StrEnum)


class YamlSidecarMetadata:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve(strict=True)

    def fingerprint(self, relative_path: str) -> str:
        digest = hashlib.sha256()
        for metadata_file in self._files(relative_path):
            name = metadata_file.relative_to(self._root).as_posix().encode("utf-8")
            digest.update(len(name).to_bytes(4, "big"))
            digest.update(name)
            stat = metadata_file.stat()
            # 超過上限的檔案不讀內容，用大小與修改時間代表；錯誤留到處理時回報。
            contents = (
                metadata_file.read_bytes()
                if stat.st_size <= MAX_METADATA_FILE_BYTES
                else f"oversized:{stat.st_size}:{stat.st_mtime_ns}".encode()
            )
            digest.update(len(contents).to_bytes(8, "big"))
            digest.update(contents)
        return digest.hexdigest()

    def declared(self, relative_path: str) -> DeclaredMetadata:
        merged: dict[str, Any] = {}
        for metadata_file in self._files(relative_path):
            merged = _deep_merge(merged, _read_mapping(metadata_file, self._root))
        unknown = set(merged) - _KNOWN_FIELDS
        if unknown:
            raise InvalidMetadataError(f"unknown metadata fields: {', '.join(sorted(unknown))}")
        return _declared(merged)

    def _files(self, relative_path: str) -> tuple[Path, ...]:
        relative = PurePosixPath(relative_path)
        folders = [self._root]
        for part in relative.parent.parts:
            folders.append(folders[-1] / part)
        candidates = [folder / FOLDER_METADATA_FILE for folder in folders]
        candidates.append(self._root / relative.parent / (relative.name + FILE_METADATA_SUFFIX))
        return tuple(path for path in candidates if path.is_file() and not path.is_symlink())


def _read_mapping(path: Path, root: Path) -> dict[str, Any]:
    name = path.relative_to(root).as_posix()
    if path.stat().st_size > MAX_METADATA_FILE_BYTES:
        raise InvalidMetadataError(f"{name} exceeds {MAX_METADATA_FILE_BYTES} bytes")
    try:
        value = yaml.safe_load(path.read_bytes().decode("utf-8-sig"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise InvalidMetadataError(f"{name} is not valid UTF-8 YAML") from exc
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise InvalidMetadataError(f"{name} must contain a mapping with string keys")
    if "metadata" in value:
        if set(value) != {"metadata"} or not isinstance(value["metadata"], dict):
            raise InvalidMetadataError(f"{name} may only nest fields under a single 'metadata' key")
        value = value["metadata"]
    return dict(value)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        result[key] = (
            _deep_merge(existing, value)
            if isinstance(existing, dict) and isinstance(value, dict)
            else value
        )
    return result


def _declared(values: Mapping[str, Any]) -> DeclaredMetadata:
    acl = values.get("acl", [])
    if not isinstance(acl, list):
        raise InvalidMetadataError("acl must be a list")
    external = values.get("external_llm_allowed")
    if external is not None and not isinstance(external, bool):
        raise InvalidMetadataError("external_llm_allowed must be true or false")
    return DeclaredMetadata(
        title=_text(values.get("title"), "title"),
        document_type=_text(values.get("document_type"), "document_type"),
        owner_department=_text(values.get("owner_department"), "owner_department"),
        document_owner=_text(values.get("document_owner"), "document_owner"),
        language=_text(values.get("language"), "language"),
        status=_enum(DocumentStatus, values.get("status"), "status"),
        confidentiality=_enum(Confidentiality, values.get("confidentiality"), "confidentiality"),
        external_llm_allowed=external,
        effective_at=_date(values.get("effective_at"), "effective_at"),
        expires_at=_date(values.get("expires_at"), "expires_at"),
        access_rules=tuple(_access_rule(entry) for entry in acl),
    )


def _access_rule(value: object) -> AccessRule:
    if not isinstance(value, dict):
        raise InvalidMetadataError("each acl entry must be a mapping")
    unknown = set(value) - _ACL_FIELDS
    if unknown:
        raise InvalidMetadataError(f"unknown acl fields: {', '.join(sorted(map(str, unknown)))}")
    principal_type = _enum(PrincipalType, value.get("principal_type"), "principal_type")
    principal_id = _text(value.get("principal_id"), "principal_id")
    if principal_type is None or principal_id is None:
        raise InvalidMetadataError("acl entries need principal_type and principal_id")
    effect = _enum(PermissionEffect, value.get("effect"), "effect") or PermissionEffect.ALLOW
    try:
        return AccessRule(principal_type=principal_type, principal_id=principal_id, effect=effect)
    except DomainValidationError as exc:
        raise InvalidMetadataError(str(exc)) from exc


def _text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidMetadataError(f"{name} must be text")
    return value.strip() or None


def _enum(enum_type: type[_Enum], value: object, name: str) -> _Enum | None:
    if value is None:
        return None
    allowed = {member.value: member for member in enum_type}
    if not isinstance(value, str) or value not in allowed:
        raise InvalidMetadataError(f"{name} must be one of: {', '.join(allowed)}")
    return allowed[value]


def _date(value: object, name: str) -> date | None:
    if value is None or isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise InvalidMetadataError(f"{name} must use YYYY-MM-DD") from exc
    raise InvalidMetadataError(f"{name} must be a date")
