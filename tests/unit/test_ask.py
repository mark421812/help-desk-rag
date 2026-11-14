"""提問流程：接受與回放、依狀態機寫入每一步、失敗代碼、追問的歷史、回收中斷的回合。"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from uuid import UUID, uuid4

import pytest

from helpdesk.application.conversation.ask import Submission, request_fingerprint
from helpdesk.application.conversation.ports import ConversationNotFoundError
from helpdesk.application.llm import (
    LlmContractError,
    LlmRateLimitedError,
    LlmRequest,
    LlmResponse,
    LlmTimeoutError,
    LlmUnavailableError,
)
from helpdesk.application.retrieval.access import AclAccess
from helpdesk.application.retrieval.ports import StoredChunk
from helpdesk.domain.access import Principal
from helpdesk.domain.conversations import ConversationId, Turn, TurnOutcome, TurnState
from helpdesk.domain.errors import ArchivedConversationError, IdempotencyConflictError
from helpdesk.infrastructure.llm.fake import FakeLlmProvider, Reply
from tests.conversation_fakes import STALE_AFTER, Pipeline, answer_reply, pipeline, rewrite_reply
from tests.retrieval_fakes import RecordingScan, document, stored

OWNER = "browser:5b1f3c1e-0000-4000-8000-000000000001"
PRINCIPAL = Principal(user_id=OWNER)
EVIDENCE_TEXT = "報名時間為一一四年十二月一日至十二月十日"
ANSWER_TEXT = "報名時間是十二月一日到十日。"


def brochure() -> list[StoredChunk]:
    owner = document(title="政治大學115學年度招生簡章")
    return [stored(owner, content=EVIDENCE_TEXT, page=10)]


async def conversation_of(pipe: Pipeline, owner: str = OWNER) -> ConversationId:
    return (await pipe.conversations.create(owner, "報名")).conversation.id


async def ask(
    pipe: Pipeline,
    conversation_id: ConversationId,
    question: str = "碩士在職專班的報名時間？",
    *,
    key: UUID | None = None,
    owner: str = OWNER,
) -> Submission:
    return await pipe.submit.run(
        owner, conversation_id, idempotency_key=key or uuid4(), question=question, language="zh-TW"
    )


async def asked_and_processed(
    chunks: list[StoredChunk], *replies: Reply, **options: object
) -> tuple[Pipeline, Turn]:
    pipe = pipeline(chunks, replies=replies, **options)  # type: ignore[arg-type]
    submission = await ask(pipe, await conversation_of(pipe))
    report = await pipe.process.run(submission.turn, PRINCIPAL)
    return pipe, report.turn


async def test_a_new_turn_is_accepted_and_the_same_request_replays_it() -> None:
    pipe = pipeline(brochure())
    conversation_id = await conversation_of(pipe)
    key = uuid4()

    first = await ask(pipe, conversation_id, "  報名時間？\r\n", key=key)
    again = await ask(pipe, conversation_id, "報名時間？", key=key)

    assert first.created and first.turn.state is TurnState.ACCEPTED
    assert first.turn.original_question == "報名時間？"
    assert first.turn.request_fingerprint == request_fingerprint("報名時間？", "zh-TW")
    assert not again.created and again.turn == first.turn
    assert len(await pipe.repository.turns(conversation_id)) == 1


async def test_a_reused_key_with_another_question_conflicts() -> None:
    pipe = pipeline(brochure())
    conversation_id = await conversation_of(pipe)
    key = uuid4()
    await ask(pipe, conversation_id, "報名時間？", key=key)

    with pytest.raises(IdempotencyConflictError):
        await ask(pipe, conversation_id, "報名費？", key=key)


async def test_only_the_owner_can_ask_in_a_conversation() -> None:
    pipe = pipeline(brochure())
    conversation_id = await conversation_of(pipe)

    with pytest.raises(ConversationNotFoundError):
        await ask(pipe, conversation_id, owner="browser:someone-else")


async def test_an_archived_conversation_takes_no_new_turns_but_still_replays() -> None:
    pipe = pipeline(brochure())
    conversation_id = await conversation_of(pipe)
    key = uuid4()
    await ask(pipe, conversation_id, key=key)
    await pipe.conversations.update(OWNER, conversation_id, archive=True)

    assert not (await ask(pipe, conversation_id, key=key)).created
    with pytest.raises(ArchivedConversationError):
        await ask(pipe, conversation_id)


async def test_an_answered_turn_persists_every_state_in_order() -> None:
    pipe, turn = await asked_and_processed(brochure(), answer_reply((ANSWER_TEXT, ["C1"])))

    assert pipe.repository.states_of(turn.id) == [
        TurnState.ACCEPTED,
        TurnState.RETRIEVING,
        TurnState.GENERATING,
        TurnState.COMPLETED,
    ]
    stored_turn = await pipe.repository.turn(turn.conversation_id, turn.id)
    assert stored_turn == turn
    assert turn.result is not None and turn.result.answer is not None
    assert [part.text for part in turn.result.answer.parts] == [ANSWER_TEXT]
    assert turn.standalone_question == turn.original_question


async def test_no_evidence_completes_while_retrieving_without_the_model() -> None:
    pipe, turn = await asked_and_processed([])

    assert pipe.repository.states_of(turn.id) == [
        TurnState.ACCEPTED,
        TurnState.RETRIEVING,
        TurnState.COMPLETED,
    ]
    assert turn.result is not None
    assert turn.result.outcome is TurnOutcome.INSUFFICIENT_EVIDENCE
    assert pipe.provider.requests == []


async def test_restricted_only_evidence_returns_its_citations_without_the_model() -> None:
    private = stored(document(external_llm_allowed=False), content="受限文件的機密內容")

    pipe, turn = await asked_and_processed([private], access=AclAccess())

    assert turn.result is not None
    assert turn.result.outcome is TurnOutcome.EXTERNAL_GENERATION_BLOCKED
    assert [c.chunk_id for c in turn.result.restricted_citations] == [private.chunk.id]
    assert pipe.provider.requests == []
    assert TurnState.GENERATING not in pipe.repository.states_of(turn.id)


async def test_a_scan_block_completes_as_outbound_policy_blocked() -> None:
    pipe, turn = await asked_and_processed(brochure(), scan=RecordingScan("taiwan_national_id"))

    assert turn.result is not None
    assert turn.result.outcome is TurnOutcome.OUTBOUND_POLICY_BLOCKED
    assert pipe.repository.states_of(turn.id)[-2:] == [TurnState.GENERATING, TurnState.COMPLETED]
    assert pipe.provider.requests == []


FAILED, UNKNOWN = TurnState.FAILED, TurnState.FAILED_UNKNOWN


@pytest.mark.parametrize(
    ("error", "state", "code", "retry_after"),
    [
        (LlmRateLimitedError("429", retry_after_seconds=17), FAILED, "provider_rate_limited", 17),
        (LlmRateLimitedError("429"), FAILED, "provider_rate_limited", None),
        (LlmTimeoutError("slow", error_code="timeout"), UNKNOWN, "provider_timeout", None),
        (LlmContractError("bad", error_code="x"), FAILED, "provider_contract_error", None),
        (LlmUnavailableError("down", status=503), FAILED, "provider_unavailable", None),
        (RuntimeError("unexpected"), UNKNOWN, "processing_interrupted", None),
    ],
)
async def test_generation_failures_map_to_safe_codes(
    error: Exception, state: TurnState, code: str, retry_after: int | None
) -> None:
    pipe, turn = await asked_and_processed(brochure(), error)

    assert (turn.state, turn.error_code, turn.retry_after_seconds) == (state, code, retry_after)
    assert await pipe.repository.turn(turn.conversation_id, turn.id) == turn


async def test_an_invalid_answer_fails_as_a_contract_error() -> None:
    _, turn = await asked_and_processed(brochure(), answer_reply((ANSWER_TEXT, ["C9"])))

    assert (turn.state, turn.error_code) == (TurnState.FAILED, "provider_contract_error")


async def test_a_retrieval_failure_fails_the_turn() -> None:
    pipe = pipeline(brochure())
    pipe.vectors.error = RuntimeError("qdrant down")
    submission = await ask(pipe, await conversation_of(pipe))

    report = await pipe.process.run(submission.turn, PRINCIPAL)

    assert (report.turn.state, report.turn.error_code) == (TurnState.FAILED, "retrieval_failed")
    assert report.error_type == "RuntimeError"


async def test_a_follow_up_is_rewritten_from_this_conversations_completed_turns() -> None:
    first_answer = "需具學士學位並有工作經驗三年以上。"
    pipe = pipeline(
        brochure(),
        replies=[
            answer_reply((first_answer, ["C1"])),
            answer_reply(("另一個對話的回答。", ["C1"])),
            rewrite_reply("碩士在職專班的報名時間是什麼時候？"),
            answer_reply((ANSWER_TEXT, ["C1"])),
        ],
    )
    conversation_id = await conversation_of(pipe)
    other_id = await conversation_of(pipe)
    first = await ask(pipe, conversation_id, "碩士在職專班的報名資格是什麼？")
    await pipe.process.run(first.turn, PRINCIPAL)
    other = await ask(pipe, other_id, "另一個對話的問題")
    await pipe.process.run(other.turn, PRINCIPAL)
    follow_up = await ask(pipe, conversation_id, "那報名時間呢？")

    report = await pipe.process.run(follow_up.turn, PRINCIPAL)

    assert report.turn.standalone_question == "碩士在職專班的報名時間是什麼時候？"
    assert report.rewrite is not None and report.rewrite.value == "rewritten"
    rewrite_request: LlmRequest = pipe.provider.requests[2]
    history = rewrite_request.messages[1].content
    assert "碩士在職專班的報名資格是什麼？" in history and first_answer in history
    assert "另一個對話" not in history


async def test_failed_turns_are_not_used_as_history() -> None:
    pipe = pipeline(
        brochure(), replies=[LlmUnavailableError("down"), answer_reply((ANSWER_TEXT, ["C1"]))]
    )
    conversation_id = await conversation_of(pipe)
    failed = await ask(pipe, conversation_id, "第一個問題")
    await pipe.process.run(failed.turn, PRINCIPAL)
    second = await ask(pipe, conversation_id, "第二個問題")

    report = await pipe.process.run(second.turn, PRINCIPAL)

    assert report.rewrite is not None and report.rewrite.value == "no_history"
    assert len(pipe.provider.requests) == 2


async def test_recovery_fails_stale_open_turns_once_and_processing_then_stops() -> None:
    pipe = pipeline(brochure(), replies=[answer_reply((ANSWER_TEXT, ["C1"]))])
    conversation_id = await conversation_of(pipe)
    stale = await ask(pipe, conversation_id)
    fresh_conversation = await conversation_of(pipe)
    pipe.repository.now += STALE_AFTER + timedelta(seconds=1)
    fresh = await ask(pipe, fresh_conversation)

    assert await pipe.recover.run() == 1
    assert await pipe.recover.run() == 0
    recovered = await pipe.repository.turn(conversation_id, stale.turn.id)
    assert recovered is not None
    assert (recovered.state, recovered.error_code) == (
        TurnState.FAILED_UNKNOWN,
        "processing_interrupted",
    )
    untouched = await pipe.repository.turn(fresh_conversation, fresh.turn.id)
    assert untouched is not None and untouched.state is TurnState.ACCEPTED

    report = await pipe.process.run(stale.turn, PRINCIPAL)

    assert report.stopped
    assert pipe.provider.requests == []
    final = await pipe.repository.turn(conversation_id, stale.turn.id)
    assert final is not None and final.state is TurnState.FAILED_UNKNOWN


async def test_a_cancelled_turn_is_marked_interrupted() -> None:
    started = asyncio.Event()

    class SlowProvider(FakeLlmProvider):
        async def complete(self, request: LlmRequest) -> LlmResponse:
            started.set()
            await asyncio.sleep(60)
            raise AssertionError("the turn should have been cancelled")

    pipe = pipeline(brochure(), provider=SlowProvider())
    submission = await ask(pipe, await conversation_of(pipe))
    task = asyncio.create_task(pipe.process.run(submission.turn, PRINCIPAL))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    turn = await pipe.repository.turn(submission.turn.conversation_id, submission.turn.id)
    assert turn is not None
    assert (turn.state, turn.error_code) == (TurnState.FAILED_UNKNOWN, "processing_interrupted")
