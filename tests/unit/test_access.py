"""權限規則：AclAccess 的拒絕優先、狀態與日期；OpenAccess 全部放行且全部可外送。"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from helpdesk.application.retrieval.access import AclAccess, OpenAccess
from helpdesk.application.retrieval.ports import CurrentChunks, ReadableBy
from helpdesk.domain.access import AccessRule, PermissionEffect, Principal, PrincipalType
from helpdesk.domain.documents import DocumentStatus
from tests.retrieval_fakes import TODAY, document, stored

ALICE = Principal(user_id="alice", group_ids=("staff",))
BOB = Principal(user_id="bob")
STAFF = AccessRule(PrincipalType.GROUP, "staff")
DENY_ALICE = AccessRule(PrincipalType.USER, "alice", PermissionEffect.DENY)
DENY_STAFF = AccessRule(PrincipalType.GROUP, "staff", PermissionEffect.DENY)


@pytest.mark.parametrize(
    ("kwargs", "principal", "readable"),
    [
        ({}, ALICE, True),
        ({"status": DocumentStatus.DRAFT}, ALICE, False),
        ({"status": DocumentStatus.EXPIRED}, ALICE, False),
        ({"effective_at": date(2026, 11, 14)}, ALICE, False),
        ({"effective_at": TODAY}, ALICE, True),
        ({"expires_at": date(2026, 11, 12)}, ALICE, False),
        ({"expires_at": TODAY}, ALICE, True),
        ({"rules": (STAFF,)}, ALICE, True),
        ({"rules": (STAFF,)}, BOB, False),
        ({"rules": (AccessRule(PrincipalType.USER, "bob"),)}, BOB, True),
        ({"rules": (STAFF, DENY_ALICE)}, ALICE, False),
        ({"rules": (DENY_STAFF,)}, ALICE, False),
        ({"rules": (DENY_STAFF,)}, BOB, True),
    ],
)
def test_acl_access_admits_by_status_dates_and_rules_with_deny_winning(
    kwargs: dict[str, Any], principal: Principal, readable: bool
) -> None:
    candidate = stored(document(**kwargs))

    assert AclAccess().admits(candidate, principal, TODAY) is readable


def test_open_access_ignores_acl_status_and_dates() -> None:
    blocked = stored(
        document(
            status=DocumentStatus.DRAFT,
            expires_at=date(2020, 1, 1),
            rules=(STAFF, DENY_ALICE),
        )
    )

    assert OpenAccess().admits(blocked, ALICE, TODAY)


def test_only_acl_access_respects_external_llm_allowed() -> None:
    private = document(external_llm_allowed=False)

    assert AclAccess().sendable(private) is False
    assert AclAccess().sendable(document()) is True
    assert OpenAccess().sendable(private) is True


def test_each_policy_produces_its_vector_scope() -> None:
    assert OpenAccess().vector_scope(ALICE) == CurrentChunks()
    assert AclAccess().vector_scope(ALICE) == ReadableBy(ALICE)
