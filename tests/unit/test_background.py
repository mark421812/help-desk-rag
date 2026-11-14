"""背景回合：同時處理的回合數不超過上限，API 啟動時把上一個程序留下的中斷回合標成失敗。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import timedelta
from typing import Any
from uuid import uuid4

from helpdesk.application.conversation.ask import Submission
from helpdesk.application.llm import LlmRequest, LlmResponse
from helpdesk.domain.access import Principal
from helpdesk.domain.conversations import TurnState
from helpdesk.infrastructure.llm.fake import FakeLlmProvider
from helpdesk.presentation.api.background import TurnRunner
from helpdesk.presentation.api.middleware import BROWSER_ID_COOKIE
from tests.conversation_fakes import (
    STALE_AFTER,
    Pipeline,
    answer_reply,
    api_runtime,
    client,
    pipeline,
)
from tests.retrieval_fakes import document, stored

BROWSER = uuid4()
OWNER = f"browser:{BROWSER}"
PRINCIPAL = Principal(user_id=OWNER)
EVIDENCE_TEXT = "報名時間為十二月一日至十日"
ANSWER = answer_reply(("報名時間是十二月一日到十日。", ["C1"]))


class GatedProvider(FakeLlmProvider):
    """每次呼叫都停在閘門前，記下同時有幾個呼叫在等。"""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()
        self.running = 0
        self.peak = 0

    async def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            await self.gate.wait()
        finally:
            self.running -= 1
        return LlmResponse(content=ANSWER, finish_reason="stop", model="fake")


def brochure_pipeline(provider: FakeLlmProvider | None = None) -> Pipeline:
    owner = document(title="政治大學115學年度招生簡章")
    return pipeline([stored(owner, content=EVIDENCE_TEXT)], provider=provider)


async def accepted_turn(pipe: Pipeline) -> Submission:
    conversation = await pipe.conversations.create(OWNER, "報名")
    return await pipe.submit.run(
        OWNER,
        conversation.conversation.id,
        idempotency_key=uuid4(),
        question="碩士在職專班的報名時間？",
        language="zh-TW",
    )


async def settle(condition: Callable[[], bool]) -> None:
    for _ in range(1_000):
        if condition():
            return
        await asyncio.sleep(0)
    raise AssertionError("the condition never held")


async def test_no_more_turns_run_at_once_than_the_concurrency_limit() -> None:
    provider = GatedProvider()
    pipe = brochure_pipeline(provider)
    runner = TurnRunner(pipe.process, pipe.recover, max_concurrency=2)
    submissions = [await accepted_turn(pipe) for _ in range(4)]

    for submission in submissions:
        runner.submit(submission.turn, PRINCIPAL)
    await settle(lambda: provider.running == 2)
    for _ in range(50):
        await asyncio.sleep(0)

    def states() -> list[TurnState]:
        return [pipe.repository.states_of(item.turn.id)[-1] for item in submissions]

    assert provider.running == 2
    assert sorted(states()) == sorted([TurnState.GENERATING] * 2 + [TurnState.ACCEPTED] * 2)

    provider.gate.set()
    await settle(lambda: states() == [TurnState.COMPLETED] * 4)
    await runner.close()

    assert provider.peak == 2 and len(provider.requests) == 4


def test_api_startup_fails_turns_a_previous_process_left_open() -> None:
    pipe = brochure_pipeline(FakeLlmProvider([ANSWER]))
    left_open = asyncio.run(accepted_turn(pipe))
    pipe.repository.now += STALE_AFTER + timedelta(seconds=1)

    with client(api_runtime(pipe)) as http:
        http.cookies.set(BROWSER_ID_COOKIE, str(BROWSER))
        status_url = (
            f"/api/v1/conversations/{left_open.turn.conversation_id}/turns/{left_open.turn.id}"
        )
        deadline = time.monotonic() + 5
        turn: dict[str, Any] = http.get(status_url).json()
        while turn["state"] == "accepted" and time.monotonic() < deadline:
            time.sleep(0.01)
            turn = http.get(status_url).json()

    assert turn["state"] == "failed_unknown"
    assert turn["error"] == {"code": "processing_interrupted", "retry_after_seconds": None}
    assert turn["answer"] is None and pipe.provider.requests == []
