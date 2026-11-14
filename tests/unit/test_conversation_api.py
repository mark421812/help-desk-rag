"""對話與提問的 HTTP 行為：202 與輪詢、回放與衝突、擁有者範圍、瀏覽器 cookie、網頁與 CSP、log。

每個請求都走正式的 app、middleware 與背景回合，只有儲存與模型換成記憶體替身。
"""

from __future__ import annotations

import json
import time
from typing import Any
from uuid import UUID, uuid4

import pytest
import structlog
from fastapi.testclient import TestClient

from helpdesk.application.llm import LlmRateLimitedError
from helpdesk.application.retrieval.access import AclAccess
from helpdesk.application.retrieval.ports import ReadableBy, StoredChunk
from helpdesk.domain.access import Principal
from helpdesk.infrastructure.identity import DevelopmentHeaderIdentity
from helpdesk.infrastructure.llm.fake import Reply
from helpdesk.presentation.api.middleware import BROWSER_ID_COOKIE
from tests.conversation_fakes import Pipeline, answer_reply, api_runtime, client, pipeline
from tests.retrieval_fakes import document, stored

QUESTION = "碩士在職專班的報名時間是什麼時候？"
EVIDENCE_TEXT = "報名時間為一一四年十二月一日至十二月十日"
ANSWER_TEXT = "報名時間是十二月一日到十日。"
FEE_TEXT = "報名費一千五百元。"
OPEN = {"accepted", "retrieving", "generating"}


def brochure() -> list[StoredChunk]:
    owner = document(title="政治大學115學年度招生簡章")
    return [stored(owner, content=EVIDENCE_TEXT, page=10), stored(owner, page=11)]


def answering(*replies: Reply, chunks: list[StoredChunk] | None = None) -> Pipeline:
    return pipeline(brochure() if chunks is None else chunks, replies=replies)


def create(http: TestClient, title: str | None = "報名") -> str:
    response = http.post("/api/v1/conversations", json={"title": title})
    assert response.status_code == 201
    conversation_id: str = response.json()["conversation_id"]
    return conversation_id


def submit(
    http: TestClient, conversation_id: str, question: str = QUESTION, key: UUID | None = None
) -> Any:
    return http.post(
        f"/api/v1/conversations/{conversation_id}/turns",
        json={"question": question, "language": "zh-TW"},
        headers={"Idempotency-Key": str(key or uuid4())},
    )


def poll(http: TestClient, status_url: str) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        body: dict[str, Any] = http.get(status_url).json()
        seen.append(body)
        if body["state"] not in OPEN:
            return seen
        time.sleep(0.01)
    raise AssertionError("the turn did not reach a terminal state")


def test_a_question_is_accepted_then_polled_until_the_cited_answer_appears() -> None:
    pipe = answering(answer_reply((ANSWER_TEXT, ["C1"]), (FEE_TEXT, ["C2", "C1"])))

    with client(api_runtime(pipe)) as http:
        conversation_id = create(http)
        accepted = submit(http, conversation_id)
        seen = poll(http, accepted.json()["status_url"])

    assert accepted.status_code == 202
    body = accepted.json()
    assert accepted.headers["location"] == body["status_url"]
    assert body["status_url"] == f"/api/v1/conversations/{conversation_id}/turns/{body['turn_id']}"
    assert body["state"] in OPEN and body["answer"] is None
    assert all(item["answer"] is None for item in seen if item["state"] in OPEN)
    final = seen[-1]
    assert (final["state"], final["outcome"], final["error"]) == ("completed", "answered", None)
    assert final["answer"] == {
        "parts": [{"text": ANSWER_TEXT, "citations": [1]}, {"text": FEE_TEXT, "citations": [2, 1]}],
        "citations": [
            {"number": 1, "title": "政治大學115學年度招生簡章", "section_path": "報名", "page": 10},
            {"number": 2, "title": "政治大學115學年度招生簡章", "section_path": "報名", "page": 11},
        ],
    }
    assert final["question"] == QUESTION and final["standalone_question"] == QUESTION


def test_the_same_key_replays_the_turn_and_another_question_conflicts() -> None:
    pipe = answering(answer_reply((ANSWER_TEXT, ["C1"])))
    key = uuid4()

    with client(api_runtime(pipe)) as http:
        conversation_id = create(http)
        first = submit(http, conversation_id, key=key)
        poll(http, first.json()["status_url"])
        replay = submit(http, conversation_id, key=key)
        conflict = submit(http, conversation_id, "報名費多少？", key=key)
        turns = http.get(f"/api/v1/conversations/{conversation_id}/turns").json()["turns"]

    assert (first.status_code, replay.status_code) == (202, 200)
    assert replay.json()["turn_id"] == first.json()["turn_id"]
    assert replay.json()["outcome"] == "answered"
    assert (conflict.status_code, conflict.json()["error"]) == (409, "idempotency_conflict")
    assert len(turns) == 1 and len(pipe.provider.requests) == 1


@pytest.mark.parametrize("headers", [{}, {"Idempotency-Key": "not-a-uuid"}])
def test_a_turn_needs_a_uuid_idempotency_key(headers: dict[str, str]) -> None:
    with client(api_runtime(answering())) as http:
        conversation_id = create(http)
        response = http.post(
            f"/api/v1/conversations/{conversation_id}/turns",
            json={"question": QUESTION},
            headers=headers,
        )

    assert (response.status_code, response.json()["error"]) == (422, "invalid_request")


def test_another_browser_gets_404_for_every_conversation_route() -> None:
    pipe = answering(answer_reply((ANSWER_TEXT, ["C1"])))

    with client(api_runtime(pipe)) as http:
        conversation_id = create(http)
        turn = submit(http, conversation_id).json()
        poll(http, turn["status_url"])
        http.cookies.clear()
        base = f"/api/v1/conversations/{conversation_id}"
        responses = [
            http.get(base),
            http.get(f"{base}/turns"),
            http.get(turn["status_url"]),
            submit(http, conversation_id),
            http.patch(base, json={"title": "搶走"}),
            http.patch(base, json={"status": "archived"}),
        ]
        listed = http.get("/api/v1/conversations").json()

    assert [(r.status_code, r.json()["error"]) for r in responses] == [(404, "not_found")] * 6
    assert listed == {"conversations": [], "next_cursor": None}


def test_the_first_visit_sets_an_http_only_lax_cookie_that_identifies_the_browser() -> None:
    with client(api_runtime(answering())) as http:
        first = http.get("/")
        again = http.get("/api/v1/conversations")

    cookie = first.headers["set-cookie"]
    name, _, rest = cookie.partition("=")
    value = rest.split(";")[0]
    assert name == BROWSER_ID_COOKIE
    assert str(UUID(value)) == value
    attributes = {part.strip() for part in cookie.split(";")[1:]}
    assert {"HttpOnly", "SameSite=Lax", "Path=/"} <= attributes
    assert "Secure" not in attributes
    assert "set-cookie" not in again.headers


def test_production_marks_the_browser_cookie_secure() -> None:
    with client(api_runtime(answering()), app_env="production") as http:
        cookie = http.get("/").headers["set-cookie"]

    assert "Secure" in {part.strip() for part in cookie.split(";")}


def test_the_web_page_and_assets_are_served_with_a_same_origin_csp() -> None:
    with client(api_runtime(answering())) as http:
        page = http.get("/")
        script = http.get("/static/app.js")
        styles = http.get("/static/styles.css")

    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert '<script src="/static/app.js" defer></script>' in page.text
    assert "<style" not in page.text and " style=" not in page.text
    for response in (page, script, styles):
        assert response.status_code == 200
        policy = response.headers["content-security-policy"]
        assert "default-src 'self'" in policy and "script-src 'self'" in policy
        assert "frame-ancestors 'none'" in policy and "unsafe" not in policy
        assert response.headers["x-content-type-options"] == "nosniff"
    assert script.headers["content-type"].startswith(("text/javascript", "application/javascript"))
    assert "產生回答中" in script.text and "請稍候再試" in script.text


def test_conversations_list_newest_activity_first_in_keyset_pages() -> None:
    pipe = answering(answer_reply((ANSWER_TEXT, ["C1"])))

    with client(api_runtime(pipe)) as http:
        oldest, middle, newest = (create(http, title) for title in ("一", "二", "三"))
        poll(http, submit(http, oldest).json()["status_url"])
        first = http.get("/api/v1/conversations", params={"limit": 2}).json()
        second = http.get(
            "/api/v1/conversations", params={"limit": 2, "cursor": first["next_cursor"]}
        ).json()
        broken = http.get("/api/v1/conversations", params={"cursor": "not-a-cursor"})

    def ids(page: dict[str, Any]) -> list[str]:
        return [item["conversation_id"] for item in page["conversations"]]

    assert ids(first) == [oldest, newest] and first["next_cursor"]
    assert ids(second) == [middle] and second["next_cursor"] is None
    assert (broken.status_code, broken.json()["error"]) == (422, "invalid_request")


def test_a_conversation_can_be_renamed_and_archived() -> None:
    with client(api_runtime(answering())) as http:
        conversation_id = create(http, None)
        base = f"/api/v1/conversations/{conversation_id}"
        renamed = http.patch(base, json={"title": "  報名問題  "})
        empty = http.patch(base, json={})
        unarchive = http.patch(base, json={"status": "active"})
        archived = http.patch(base, json={"status": "archived"})
        listed = http.get("/api/v1/conversations").json()
        refused = submit(http, conversation_id)
        still_readable = http.get(base)

    assert renamed.json()["title"] == "報名問題"
    assert empty.status_code == 422 and unarchive.status_code == 422
    assert archived.json()["status"] == "archived"
    assert listed["conversations"] == []
    assert (refused.status_code, refused.json()["error"]) == (409, "conversation_archived")
    assert still_readable.json()["title"] == "報名問題"


def test_a_rate_limited_turn_reports_how_long_to_wait() -> None:
    pipe = answering(LlmRateLimitedError("429", retry_after_seconds=17))

    with client(api_runtime(pipe)) as http:
        final = poll(http, submit(http, create(http)).json()["status_url"])[-1]

    assert final["state"] == "failed"
    assert final["error"] == {"code": "provider_rate_limited", "retry_after_seconds": 17}
    assert final["answer"] is None and final["outcome"] is None


def test_restricted_only_evidence_lists_citations_without_content() -> None:
    private = stored(document(external_llm_allowed=False), content="受限文件的機密內容", page=7)
    pipe = pipeline([private], access=AclAccess())

    with client(api_runtime(pipe, identity=DevelopmentHeaderIdentity())) as http:
        http.headers["x-dev-user-id"] = "alice"
        final = poll(http, submit(http, create(http)).json()["status_url"])[-1]
        text = json.dumps(final, ensure_ascii=False)

    assert final["outcome"] == "external_generation_blocked"
    assert [item["page"] for item in final["restricted_citations"]] == [7]
    assert "機密內容" not in text and pipe.provider.requests == []


def test_access_control_scopes_conversations_to_the_development_user() -> None:
    pipe = pipeline(brochure(), replies=[answer_reply((ANSWER_TEXT, ["C1"]))], access=AclAccess())

    with client(api_runtime(pipe, identity=DevelopmentHeaderIdentity())) as http:
        missing = http.post("/api/v1/conversations", json={})
        http.headers.update({"x-dev-user-id": "alice", "x-dev-group-ids": "staff"})
        conversation_id = create(http)
        poll(http, submit(http, conversation_id).json()["status_url"])
        http.headers.update({"x-dev-user-id": "bob"})
        foreign = http.get(f"/api/v1/conversations/{conversation_id}")

    assert (missing.status_code, missing.json()["error"]) == (401, "unauthorized")
    assert pipe.vectors.scopes[-1] == ReadableBy(Principal(user_id="alice", group_ids=("staff",)))
    assert foreign.status_code == 404


def test_logs_carry_codes_but_never_questions_answers_evidence_or_identities() -> None:
    pipe = answering(answer_reply((ANSWER_TEXT, ["C1"])))

    with structlog.testing.capture_logs() as events, client(api_runtime(pipe)) as http:
        http.headers["x-dev-user-id"] = "alice@example.com"
        conversation_id = create(http, "私人標題")
        poll(http, submit(http, conversation_id).json()["status_url"])
        browser_id = http.cookies[BROWSER_ID_COOKIE]

    logged = json.dumps(events, ensure_ascii=False, default=str)
    [processed] = [event for event in events if event["event"] == "turn_processed"]
    assert (processed["state"], processed["outcome"], processed["sendable"]) == (
        "completed",
        "answered",
        2,
    )
    for text in (QUESTION, ANSWER_TEXT, EVIDENCE_TEXT, "私人標題", browser_id, "alice", "報名"):
        assert text not in logged
