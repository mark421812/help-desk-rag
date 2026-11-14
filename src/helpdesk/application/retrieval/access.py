"""權限過濾：`ACCESS_CONTROL_ENABLED` 在 composition root 選定其中一個實作。

`AclAccess` 有兩道關卡。向量檢索先用 `ReadableBy` 在 Qdrant 預先過濾，向量與關鍵字兩路的候選
再用 `admits` 以 PostgreSQL 的最新 metadata 複驗，兩路走同一份規則。拒絕永遠優先於允許；沒有
任何允許規則的文件視為公開，和寫進 Qdrant 的 `acl_public` 一致。

`OpenAccess` 不比對 ACL、文件狀態與生效、到期日，所有文件都可外送。兩種模式都只採用目前版本
的 chunk，這一點由粗檢索負責，不在這裡。
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from helpdesk.application.retrieval.ports import (
    CurrentChunks,
    ReadableBy,
    StoredChunk,
    VectorScope,
)
from helpdesk.domain.access import AccessRule, PermissionEffect, Principal, PrincipalType
from helpdesk.domain.documents import Document, DocumentMetadata, DocumentStatus


class AccessPolicy(Protocol):
    def vector_scope(self, principal: Principal) -> VectorScope: ...

    def admits(self, stored: StoredChunk, principal: Principal, today: date) -> bool: ...

    def sendable(self, document: Document) -> bool:
        """這份文件的內容能不能送到外部 LLM。"""
        ...


class OpenAccess:
    def vector_scope(self, principal: Principal) -> VectorScope:
        return CurrentChunks()

    def admits(self, stored: StoredChunk, principal: Principal, today: date) -> bool:
        return True

    def sendable(self, document: Document) -> bool:
        return True


class AclAccess:
    def vector_scope(self, principal: Principal) -> VectorScope:
        return ReadableBy(principal)

    def admits(self, stored: StoredChunk, principal: Principal, today: date) -> bool:
        return readable(stored.document.metadata, principal, today)

    def sendable(self, document: Document) -> bool:
        return document.metadata.external_llm_allowed


def readable(metadata: DocumentMetadata, principal: Principal, today: date) -> bool:
    if metadata.status is not DocumentStatus.PUBLISHED:
        return False
    if metadata.effective_at is not None and metadata.effective_at > today:
        return False
    if metadata.expires_at is not None and metadata.expires_at < today:
        return False
    applicable = [rule for rule in metadata.access_rules if _applies_to(rule, principal)]
    if any(rule.effect is PermissionEffect.DENY for rule in applicable):
        return False
    if not any(rule.effect is PermissionEffect.ALLOW for rule in metadata.access_rules):
        return True
    return any(rule.effect is PermissionEffect.ALLOW for rule in applicable)


def _applies_to(rule: AccessRule, principal: Principal) -> bool:
    if rule.principal_type is PrincipalType.USER:
        return rule.principal_id == principal.user_id
    return rule.principal_id in principal.group_ids
