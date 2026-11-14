"""helpdesk-api 的 HTTP 行為：retrieve 回應、健康檢查、身分、請求上限、安全 header、log 不含內容。

每個請求都走正式的 app 與 middleware，只有儲存與模型換成記憶體替身。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import pytest
import structlog

from helpdesk.application.identity import IdentityResolver
from helpdesk.application.retrieval.access import AccessPolicy, AclAccess
from helpdesk.application.retrieval.ports import ReadableBy, StoredChunk
from helpdesk.domain.access import Principal
from helpdesk.infrastructure.identity import DevelopmentHeaderIdentity
from helpdesk.presentation.api.app import MAX_REQUEST_BYTES
from helpdesk.presentation.api.composition import ApiRuntime
from tests.conversation_fakes import api_runtime, client, pipeline, yes
from tests.retrieval_fakes import FakeVectors, document, stored

QUESTION = "那報名時間呢？"
REWRITTEN = "碩士在職專班的報名時間是什麼時候？"
HISTORY_TEXT = "需具學士學位並有工作經驗三年以上"
EVIDENCE_TEXT = "報名時間為一一四年十二月一日至十二月十日"
SECRET_TEXT = "受限文件的內容不可回傳"


async def no() -> bool:
    return False


def runtime(
    chunks: Sequence[StoredChunk],
    *,
    access: AccessPolicy | None = None,
    identity: IdentityResolver | None = None,
    replies: Sequence[str] = (),
    keyword_error: Exception | None = None,
    readiness: dict[str, Callable[[], Awaitable[bool]]] | None = None,
) -> tuple[ApiRuntime, FakeVectors]:
    pipe = pipeline(chunks, replies=replies, access=access, keyword_error=keyword_error)
    return api_runtime(pipe, identity=identity, readiness=readiness), pipe.vectors


def brochure_chunks() -> list[StoredChunk]:
    brochure = document(title="政治大學115學年度招生簡章")
    return [stored(brochure, content=EVIDENCE_TEXT, page=10), stored(brochure, page=11)]


def test_retrieve_returns_the_plan_and_evidence_with_citations_scores_and_buckets() -> None:
    chunks = brochure_chunks()
    api, _ = runtime(chunks)

    with client(api) as http:
        response = http.post("/api/v1/retrieve", json={"question": "報名時間", "language": "zh-TW"})

    assert response.status_code == 200
    body = response.json()
    assert body["standalone_question"] == "報名時間"
    assert (body["rewrite"], body["status"], body["warnings"]) == (
        "no_history",
        "evidence_found",
        [],
    )
    first = body["evidence"][0]
    assert first["rank"] == 1 and first["bucket"] == "sendable"
    assert first["content"] == EVIDENCE_TEXT
    assert first["citation"] == {
        "title": "政治大學115學年度招生簡章",
        "section_path": "報名",
        "page": 10,
    }
    assert set(first["scores"]) == {"fused", "vector", "keyword"}
    assert first["scores"]["vector"] is not None and first["scores"]["keyword"] is not None


def test_restricted_evidence_is_returned_as_a_citation_without_content() -> None:
    private = stored(document(external_llm_allowed=False), content=SECRET_TEXT)
    api, _ = runtime([private], access=AclAccess(), identity=DevelopmentHeaderIdentity())

    with client(api) as http:
        response = http.post(
            "/api/v1/retrieve", json={"question": "受限文件"}, headers={"x-dev-user-id": "alice"}
        )

    body = response.json()
    assert body["status"] == "external_generation_blocked"
    assert [(item["bucket"], item["content"]) for item in body["evidence"]] == [
        ("restricted", None)
    ]
    assert SECRET_TEXT not in response.text


def test_a_follow_up_is_rewritten_from_history() -> None:
    api, _ = runtime(
        brochure_chunks(),
        replies=[json.dumps({"standalone_question": REWRITTEN}, ensure_ascii=False)],
    )

    with client(api) as http:
        body = http.post(
            "/api/v1/retrieve",
            json={
                "question": QUESTION,
                "history": [
                    {"role": "user", "text": "報名資格是什麼？"},
                    {"role": "assistant", "text": HISTORY_TEXT},
                ],
            },
        ).json()

    assert (body["original_question"], body["standalone_question"], body["rewrite"]) == (
        QUESTION,
        REWRITTEN,
        "rewritten",
    )


def test_a_keyword_failure_is_reported_as_a_warning() -> None:
    api, _ = runtime(brochure_chunks(), keyword_error=RuntimeError("db down"))

    with client(api) as http:
        body = http.post("/api/v1/retrieve", json={"question": "報名時間"}).json()

    assert body["warnings"] == ["keyword_search_unavailable"]
    assert body["evidence"]


def test_access_control_requires_a_development_identity_and_passes_it_on() -> None:
    api, vectors = runtime(
        brochure_chunks(), access=AclAccess(), identity=DevelopmentHeaderIdentity()
    )

    with client(api) as http:
        missing = http.post("/api/v1/retrieve", json={"question": "報名時間"})
        bearer = http.post(
            "/api/v1/retrieve",
            json={"question": "報名時間"},
            headers={"authorization": "Bearer token", "x-dev-user-id": "alice"},
        )
        allowed = http.post(
            "/api/v1/retrieve",
            json={"question": "報名時間"},
            headers={"x-dev-user-id": " alice ", "x-dev-group-ids": "staff, hr"},
        )

    assert (missing.status_code, missing.json()["error"]) == (401, "unauthorized")
    assert bearer.status_code == 401
    assert allowed.status_code == 200
    assert vectors.scopes[-1] == ReadableBy(Principal(user_id="alice", group_ids=("staff", "hr")))


@pytest.mark.parametrize(
    "body",
    [
        {"question": ""},
        {"question": "q", "extra": True},
        {"question": "q", "history": [{"role": "system", "text": "x"}]},
        {"question": "q", "language": "   "},
        {"question": "q" * 20_001},
    ],
)
def test_invalid_requests_are_rejected_without_echoing_input(body: dict[str, Any]) -> None:
    api, _ = runtime([])

    with client(api) as http:
        response = http.post("/api/v1/retrieve", json=body)

    assert (response.status_code, response.json()["error"]) == (422, "invalid_request")
    assert "q" * 50 not in response.text


def test_a_body_over_the_limit_is_refused_before_parsing() -> None:
    api, _ = runtime([])
    oversized = json.dumps({"question": "q", "padding": "x" * MAX_REQUEST_BYTES})

    with client(api) as http:
        response = http.post(
            "/api/v1/retrieve", content=oversized, headers={"content-type": "application/json"}
        )

    assert (response.status_code, response.json()["error"]) == (413, "request_too_large")


def test_responses_carry_security_headers_and_the_request_id() -> None:
    api, _ = runtime([])

    with client(api) as http:
        response = http.get("/health/live", headers={"x-request-id": "req-123"})

    assert response.json() == {"status": "live"}
    assert response.headers["x-request-id"] == "req-123"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["cache-control"] == "no-store"


def test_readiness_reports_each_dependency() -> None:
    ready, _ = runtime([])
    not_ready, _ = runtime([], readiness={"postgres": yes, "qdrant": no, "embedding_model": yes})

    with client(ready) as http:
        good = http.get("/health/ready")
    with client(not_ready) as http:
        bad = http.get("/health/ready")

    assert (good.status_code, good.json()["status"]) == (200, "ready")
    assert bad.status_code == 503
    assert bad.json()["checks"] == {"postgres": True, "qdrant": False, "embedding_model": True}


def test_logs_never_contain_question_history_rewrite_or_evidence_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    api, _ = runtime(
        brochure_chunks(),
        replies=[json.dumps({"standalone_question": REWRITTEN}, ensure_ascii=False)],
        keyword_error=RuntimeError(f"SELECT ... WHERE content ILIKE '%{QUESTION}%'"),
    )
    caplog.set_level(logging.DEBUG)

    with structlog.testing.capture_logs() as events, client(api) as http:
        http.post(
            "/api/v1/retrieve",
            json={
                "question": QUESTION,
                "history": [{"role": "assistant", "text": HISTORY_TEXT}],
            },
        )
        http.post("/api/v1/retrieve", json={"question": QUESTION, "extra": 1})

    logged = json.dumps(events, ensure_ascii=False, default=str) + caplog.text
    completed = [event for event in events if event["event"] == "retrieval_completed"]
    assert completed and completed[0]["rewrite"] == "rewritten"
    assert completed[0]["keyword_failure"] == "RuntimeError"
    for text in (QUESTION, HISTORY_TEXT, REWRITTEN, EVIDENCE_TEXT, "報名"):
        assert text not in logged


def test_an_unexpected_error_returns_500_and_logs_only_its_type() -> None:
    api, vectors = runtime(brochure_chunks())
    vectors.error = RuntimeError(f"query_points failed for {QUESTION}")

    with structlog.testing.capture_logs() as events, client(api) as http:
        response = http.post("/api/v1/retrieve", json={"question": QUESTION})

    assert (response.status_code, response.json()["error"]) == (500, "internal_error")
    [failure] = [event for event in events if event["event"] == "unhandled_exception"]
    assert failure["error_type"] == "RuntimeError"
    assert failure["frames"]
    assert QUESTION not in json.dumps(events, ensure_ascii=False, default=str)
