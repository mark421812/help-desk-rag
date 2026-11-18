"""理解問題：正規化、只在有歷史時改寫、任何失敗都退回原問題。"""

from __future__ import annotations

import asyncio
import json

import pytest

from helpdesk.application.llm import (
    LlmContractError,
    LlmRateLimitedError,
    LlmRequest,
    LlmResponse,
    LlmTimeoutError,
)
from helpdesk.application.outbound_scan import OutboundScan
from helpdesk.application.retrieval.understanding import (
    HistoryMessage,
    RewriteLimits,
    RewriteOutcome,
    UnderstandQuestion,
)
from helpdesk.domain.errors import DomainValidationError
from helpdesk.infrastructure.llm.fake import FakeLlmProvider, Reply
from helpdesk.infrastructure.outbound_scan.scanners import RuleBasedOutboundScan
from tests.retrieval_fakes import RecordingScan

LIMITS = RewriteLimits(
    history_turns=3,
    turn_max_chars=500,
    max_chars=500,
    timeout_seconds=1.0,
    max_prompt_bytes=48_000,
    reasoning_effort="low",
)
FOLLOW_UP = "那報名時間呢？"
STANDALONE = "碩士在職專班的報名時間是什麼時候？"
HISTORY = [
    HistoryMessage("user", "碩士在職專班的報名資格是什麼？"),
    HistoryMessage("assistant", "需具學士學位並有工作經驗。"),
]


def understand(
    *replies: Reply, scan: OutboundScan | None = None, limits: RewriteLimits = LIMITS
) -> tuple[UnderstandQuestion, FakeLlmProvider]:
    provider = FakeLlmProvider(replies)
    return UnderstandQuestion(
        provider=provider, scan=scan or RecordingScan(), limits=limits
    ), provider


def rewrite_reply(question: str) -> str:
    return json.dumps({"standalone_question": question}, ensure_ascii=False)


def sent_payload(request: LlmRequest) -> dict[str, object]:
    payload: dict[str, object] = json.loads(request.messages[1].content)
    return payload


async def test_a_question_without_history_is_never_rewritten() -> None:
    use_case, provider = understand(rewrite_reply("不該被使用"))

    result = await use_case.run(FOLLOW_UP, "zh-TW", [])

    assert result.rewrite is RewriteOutcome.NO_HISTORY
    assert result.plan.standalone_question == FOLLOW_UP
    assert provider.requests == []


async def test_history_that_is_only_whitespace_counts_as_no_history() -> None:
    use_case, provider = understand(rewrite_reply(STANDALONE))

    result = await use_case.run(FOLLOW_UP, None, [HistoryMessage("user", " \n\t ")])

    assert result.rewrite is RewriteOutcome.NO_HISTORY
    assert provider.requests == []


async def test_the_question_is_normalized_before_anything_else() -> None:
    use_case, _ = understand()
    decomposed = "Café 報名\r\n時間？  "

    result = await use_case.run(f"  {decomposed}", None, [])

    assert result.plan.original_question == "Café 報名\n時間？"
    assert result.plan.standalone_question == "Café 報名\n時間？"


@pytest.mark.parametrize("question", ["", "   \r\n  "])
async def test_a_blank_question_is_rejected(question: str) -> None:
    use_case, _ = understand()

    with pytest.raises(DomainValidationError):
        await use_case.run(question, None, [])


async def test_a_follow_up_is_rewritten_with_the_strict_schema() -> None:
    use_case, provider = understand(rewrite_reply(STANDALONE))

    result = await use_case.run(FOLLOW_UP, "zh-TW", HISTORY)

    assert result.rewrite is RewriteOutcome.REWRITTEN
    assert result.plan.original_question == FOLLOW_UP
    assert result.plan.standalone_question == STANDALONE
    [request] = provider.requests
    assert request.output_schema.name == "standalone_question"
    assert request.output_schema.schema["additionalProperties"] is False
    assert request.temperature == 0
    assert sent_payload(request) == {
        "data_classification": "untrusted_conversation_and_question",
        "requested_language": "zh-TW",
        "history": [{"role": m.role, "text": m.text} for m in HISTORY],
        "latest_question": FOLLOW_UP,
    }


async def test_only_the_latest_turns_are_sent_and_each_is_collapsed_and_cut() -> None:
    use_case, provider = understand(
        rewrite_reply(STANDALONE),
        limits=RewriteLimits(
            history_turns=2,
            turn_max_chars=5,
            max_chars=500,
            timeout_seconds=1.0,
            max_prompt_bytes=48_000,
            reasoning_effort="low",
        ),
    )
    history = [HistoryMessage("user", f"第{index}輪  問\n題很長很長") for index in range(6)]

    await use_case.run(FOLLOW_UP, None, history)

    sent = sent_payload(provider.requests[0])["history"]
    assert sent == [{"role": "user", "text": f"第{index}輪 問"} for index in range(2, 6)]


async def test_the_scan_sees_the_question_the_language_and_every_history_text() -> None:
    scan = RecordingScan()
    use_case, _ = understand(rewrite_reply(STANDALONE), scan=scan)

    await use_case.run(FOLLOW_UP, "zh-TW", HISTORY)

    assert scan.scanned == [[FOLLOW_UP, "zh-TW", *(m.text for m in HISTORY)]]


async def test_a_scan_block_skips_the_rewrite_instead_of_failing() -> None:
    use_case, provider = understand(
        rewrite_reply(STANDALONE), scan=RuleBasedOutboundScan.packaged()
    )
    history = [*HISTORY, HistoryMessage("user", "我的 password=hunter2hunter2")]

    result = await use_case.run(FOLLOW_UP, None, history)

    assert result.rewrite is RewriteOutcome.SCAN_BLOCKED
    assert result.detail == "secret-assignment"
    assert result.plan.standalone_question == FOLLOW_UP
    assert provider.requests == []


async def test_a_rewrite_that_takes_too_long_falls_back_to_the_question() -> None:
    class SlowProvider(FakeLlmProvider):
        async def complete(self, request: LlmRequest) -> LlmResponse:
            await asyncio.sleep(5)
            raise AssertionError("the timeout should have fired")

    use_case = UnderstandQuestion(
        provider=SlowProvider(),
        scan=RecordingScan(),
        limits=RewriteLimits(
            history_turns=3,
            turn_max_chars=500,
            max_chars=500,
            timeout_seconds=0.05,
            max_prompt_bytes=48_000,
            reasoning_effort="low",
        ),
    )

    result = await use_case.run(FOLLOW_UP, None, HISTORY)

    assert result.rewrite is RewriteOutcome.TIMEOUT
    assert result.plan.standalone_question == FOLLOW_UP


@pytest.mark.parametrize(
    ("error", "outcome", "detail"),
    [
        (LlmTimeoutError("late", error_code="timeout"), RewriteOutcome.TIMEOUT, None),
        (LlmRateLimitedError("slow down"), RewriteOutcome.RATE_LIMITED, None),
        (
            LlmContractError("rejected", error_code="json_validate_failed", status=400),
            RewriteOutcome.PROVIDER_ERROR,
            "json_validate_failed",
        ),
        (RuntimeError("unexpected"), RewriteOutcome.PROVIDER_ERROR, "RuntimeError"),
    ],
)
async def test_provider_failures_fall_back_to_the_question(
    error: Exception, outcome: RewriteOutcome, detail: str | None
) -> None:
    use_case, _ = understand(error)

    result = await use_case.run(FOLLOW_UP, None, HISTORY)

    assert (result.rewrite, result.detail) == (outcome, detail)
    assert result.plan.standalone_question == FOLLOW_UP
    assert result.search_texts == (FOLLOW_UP, f"{HISTORY[0].text}\n{FOLLOW_UP}")


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        "[]",
        json.dumps({"standalone_question": STANDALONE, "extra": 1}, ensure_ascii=False),
        json.dumps({"question": STANDALONE}, ensure_ascii=False),
        json.dumps({"standalone_question": 42}),
        rewrite_reply("   "),
        rewrite_reply("第一行\n第二行"),
        rewrite_reply("第一行 第二行"),
        rewrite_reply("含有控制字元\x07"),
        rewrite_reply("長" * 501),
        LlmResponse(content=rewrite_reply(STANDALONE), finish_reason="length", model="fake"),
    ],
)
async def test_output_outside_the_contract_falls_back_to_the_question(reply: Reply) -> None:
    use_case, _ = understand(reply)

    result = await use_case.run(FOLLOW_UP, None, HISTORY)

    assert result.rewrite is RewriteOutcome.INVALID_OUTPUT
    assert result.plan.standalone_question == FOLLOW_UP
    assert result.search_texts == (FOLLOW_UP, f"{HISTORY[0].text}\n{FOLLOW_UP}")


async def test_a_rewrite_equal_to_the_question_is_reported_as_unchanged() -> None:
    use_case, _ = understand(rewrite_reply(f"  {FOLLOW_UP} "))

    result = await use_case.run(FOLLOW_UP, None, HISTORY)

    assert result.rewrite is RewriteOutcome.UNCHANGED
    assert result.plan.standalone_question == FOLLOW_UP


async def test_a_very_long_question_is_not_sent_for_rewriting() -> None:
    use_case, provider = understand(rewrite_reply(STANDALONE))

    result = await use_case.run("問" * 2_001, None, HISTORY)

    assert result.rewrite is RewriteOutcome.QUESTION_TOO_LONG
    assert provider.requests == []


async def test_a_prompt_over_the_byte_budget_is_not_sent() -> None:
    use_case, provider = understand(
        rewrite_reply(STANDALONE),
        limits=RewriteLimits(
            history_turns=3,
            turn_max_chars=500,
            max_chars=500,
            timeout_seconds=1.0,
            max_prompt_bytes=200,
            reasoning_effort="low",
        ),
    )

    result = await use_case.run(FOLLOW_UP, None, HISTORY)

    assert result.rewrite is RewriteOutcome.PROMPT_TOO_LARGE
    assert provider.requests == []


async def test_retrieval_searches_the_rewrite_when_there_is_one() -> None:
    use_case, _ = understand(rewrite_reply(STANDALONE))

    result = await use_case.run(FOLLOW_UP, None, HISTORY)

    assert result.search_texts == (STANDALONE,)
    assert result.history == tuple(HISTORY)


async def test_a_failed_rewrite_searches_with_the_most_recent_question() -> None:
    use_case, _ = understand(LlmRateLimitedError("slow down"))
    history = [
        HistoryMessage("user", "報名資格是什麼？"),
        HistoryMessage("assistant", "需具學士學位。"),
        HistoryMessage("user", "把黃小明的經歷整理給我"),
        HistoryMessage("assistant", "他有五年系統分析經驗。"),
    ]

    result = await use_case.run("用列點的方式給我重點五個點", None, history)

    assert result.search_texts == (
        "用列點的方式給我重點五個點",
        "把黃小明的經歷整理給我\n用列點的方式給我重點五個點",
    )


async def test_without_history_retrieval_searches_the_question_alone() -> None:
    use_case, _ = understand()

    result = await use_case.run(FOLLOW_UP, None, [])

    assert result.search_texts == (FOLLOW_UP,)
    assert result.history == ()
