"""composition root：兩個開關的 4 種組合各自選出行為不同的實作，ACL 開啟的 production 拒絕啟動。"""

from __future__ import annotations

from itertools import product
from typing import Any
from uuid import uuid4

import pytest

from helpdesk.application.identity import IdentityInput, UnauthorizedError
from helpdesk.application.llm import LlmUnavailableError
from helpdesk.application.retrieval.ports import CurrentChunks, ReadableBy
from helpdesk.config import Settings
from helpdesk.domain.access import Principal
from helpdesk.domain.documents import DocumentStatus
from helpdesk.presentation.api.composition import (
    MissingGroqKeyError,
    ProductionIdentityMissingError,
    build_runtime,
    llm_provider,
    policies,
)
from tests.retrieval_fakes import TODAY, FixedEmbedder, document, stored
from tests.unit.test_groq_provider import REQUEST

ALICE = Principal(user_id="alice")
BROWSER = str(uuid4())
CALLER = IdentityInput(
    development_user_id="alice", development_group_ids=("staff",), browser_id=BROWSER
)
SECRET = ["password=hunter2hunter2"]
COMBINATIONS = list(product([False, True], repeat=2))


def settings(**values: Any) -> Settings:
    return Settings(_env_file=None, llm_provider="fake", **values)


@pytest.mark.parametrize(("acl", "scan"), COMBINATIONS)
def test_each_toggle_combination_picks_its_own_policies(acl: bool, scan: bool) -> None:
    chosen = policies(settings(access_control_enabled=acl, sensitive_data_scan_enabled=scan))
    draft = stored(document(status=DocumentStatus.DRAFT, external_llm_allowed=False))

    assert chosen.access.vector_scope(ALICE) == (ReadableBy(ALICE) if acl else CurrentChunks())
    assert chosen.access.admits(draft, ALICE, TODAY) is not acl
    assert chosen.access.sendable(draft.document) is not acl
    assert chosen.scan.scan(SECRET).allowed is not scan
    expected = (
        Principal(user_id="alice", group_ids=("staff",)) if acl else Principal(f"browser:{BROWSER}")
    )
    assert chosen.identity.resolve(CALLER) == expected


@pytest.mark.parametrize(("acl", "scan"), COMBINATIONS)
async def test_the_runtime_uses_the_chosen_identity(acl: bool, scan: bool) -> None:
    runtime = build_runtime(
        settings(access_control_enabled=acl, sensitive_data_scan_enabled=scan),
        embedder=FixedEmbedder(),
    )
    try:
        resolved = runtime.identity.resolve(CALLER)
    finally:
        await runtime.close()

    assert resolved.user_id == ("alice" if acl else f"browser:{BROWSER}")


def test_access_control_off_identifies_each_browser_by_its_cookie() -> None:
    identity = policies(settings(access_control_enabled=False)).identity
    other = str(uuid4())

    assert identity.resolve(IdentityInput(browser_id=other)) != identity.resolve(CALLER)
    for broken in (IdentityInput(), IdentityInput(browser_id="not-a-uuid")):
        with pytest.raises(UnauthorizedError):
            identity.resolve(broken)


def test_access_control_on_requires_development_headers() -> None:
    identity = policies(settings(access_control_enabled=True)).identity

    with pytest.raises(UnauthorizedError):
        identity.resolve(IdentityInput(browser_id=BROWSER))


def test_access_control_on_in_production_refuses_to_start() -> None:
    production = settings(access_control_enabled=True, app_env="production")

    with pytest.raises(ProductionIdentityMissingError):
        policies(production)
    with pytest.raises(ProductionIdentityMissingError):
        build_runtime(production)


def test_access_control_off_may_run_in_production() -> None:
    assert policies(settings(access_control_enabled=False, app_env="production"))


@pytest.mark.parametrize("key", [None, "", "   "])
def test_the_api_refuses_to_start_groq_without_a_key(key: str | None) -> None:
    groq = Settings(_env_file=None, llm_provider="groq", groq_api_key=key)

    with pytest.raises(MissingGroqKeyError):
        llm_provider(groq)
    with pytest.raises(MissingGroqKeyError):
        build_runtime(groq, embedder=FixedEmbedder())


async def test_the_fake_provider_has_nothing_to_say() -> None:
    with pytest.raises(LlmUnavailableError):
        await llm_provider(settings()).complete(REQUEST)


async def test_the_runtime_reports_unreachable_stores_as_not_ready() -> None:
    runtime = build_runtime(
        settings(
            database_url="postgresql+asyncpg://nobody:nothing@127.0.0.1:9/none",
            qdrant_url="http://127.0.0.1:9",
        ),
        embedder=FixedEmbedder(),
    )
    try:
        checks = {name: await check() for name, check in runtime.readiness.items()}
    finally:
        await runtime.close()

    assert checks == {"postgres": False, "qdrant": False, "embedding_model": True}
