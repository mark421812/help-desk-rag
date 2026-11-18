"""回答模組：只有可外送的證據會送出、掃描擋下、沒有可外送證據時不叫模型、嚴格解析與引用驗證。"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import pytest

from helpdesk.application.answering.generate import (
    AnswerLimits,
    GenerateAnswer,
    PromptTooLargeError,
    settle_without_model,
)
from helpdesk.application.answering.prompt import GROUNDED_ANSWER_SCHEMA
from helpdesk.application.llm import LlmContractError, LlmRequest, LlmResponse, TokenUsage
from helpdesk.application.outbound_scan import OutboundScan
from helpdesk.application.retrieval.understanding import HistoryMessage
from helpdesk.domain.conversations import TurnOutcome
from helpdesk.domain.documents import ChunkId, DocumentId, VersionId
from helpdesk.domain.retrieval import Citation, Evidence, EvidenceSet, QueryPlan
from helpdesk.infrastructure.llm.fake import FakeLlmProvider, Reply
from tests.conversation_fakes import answer_reply
from tests.retrieval_fakes import RecordingScan

PLAN = QueryPlan(
    original_question="那報名時間呢？",
    standalone_question="碩士在職專班的報名時間是什麼時候？",
    language="zh-TW",
)
LIMITS = AnswerLimits(
    max_prompt_bytes=48_000, max_output_tokens=3_000, temperature=0.3, reasoning_effort="low"
)


def evidence(rank: int, content: str, *, page: int = 10) -> Evidence:
    return Evidence(
        rank=rank,
        citation=Citation(
            chunk_id=ChunkId(uuid4()),
            document_id=DocumentId(uuid4()),
            version_id=VersionId(uuid4()),
            title="政治大學115學年度招生簡章",
            section_path="報名",
            page_number=page,
        ),
        content=content,
        document_type="policy",
        token_count=20,
        score=1.0 / rank,
    )


DATES = evidence(1, "報名時間為十二月一日至十二月十日", page=10)
PRIVATE = evidence(2, "受限文件的機密內容", page=99)
FEES = evidence(3, "報名費新臺幣一千五百元", page=11)
MIXED = EvidenceSet(sendable=(DATES, FEES), restricted=(PRIVATE,))


def generator(
    *replies: Reply, scan: OutboundScan | None = None, limits: AnswerLimits = LIMITS
) -> tuple[GenerateAnswer, FakeLlmProvider]:
    provider = FakeLlmProvider(replies)
    return GenerateAnswer(provider=provider, scan=scan or RecordingScan(), limits=limits), provider


def sent(request: LlmRequest) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(request.messages[1].content)
    return payload


async def test_the_answer_cites_chunks_through_per_request_citation_ids() -> None:
    content = answer_reply(
        ("報名時間是十二月一日到十日。", ["C1"]), ("報名費一千五百元。", ["C2", "C1"])
    )
    generate, provider = generator(
        LlmResponse(
            content, "stop", "fake", TokenUsage(prompt=120, completion=30, cached_prompt=64)
        )
    )

    generation = await generate.run(PLAN, MIXED)

    result = generation.result
    assert result.outcome is TurnOutcome.ANSWERED
    assert result.answer is not None
    dates, fees = DATES.citation.chunk_id, FEES.citation.chunk_id
    assert [part.chunk_ids for part in result.answer.parts] == [(dates,), (fees, dates)]
    assert result.answer.citations == (DATES.citation, FEES.citation)
    assert generation.usage == TokenUsage(prompt=120, completion=30, cached_prompt=64)
    [request] = provider.requests
    assert request.output_schema == GROUNDED_ANSWER_SCHEMA
    assert (request.max_output_tokens, request.temperature) == (3_000, 0.3)
    payload = sent(request)
    assert payload["question"] == PLAN.standalone_question
    assert payload["requested_language"] == "zh-TW"
    assert [item["citation_id"] for item in payload["evidence"]] == ["C1", "C2"]


async def test_only_sendable_evidence_reaches_the_scan_and_the_model() -> None:
    scan = RecordingScan()
    generate, provider = generator(answer_reply(("報名時間是十二月。", ["C1"])), scan=scan)

    await generate.run(PLAN, MIXED)

    [request] = provider.requests
    prompt = "".join(message.content for message in request.messages)
    assert [item["content"] for item in sent(request)["evidence"]] == [DATES.content, FEES.content]
    assert PRIVATE.content not in prompt
    [scanned] = scan.scanned
    assert PLAN.standalone_question in scanned
    assert DATES.content in scanned and FEES.content in scanned
    assert PRIVATE.content not in scanned


async def test_a_scan_block_stops_before_the_model() -> None:
    generate, provider = generator(scan=RecordingScan("credential_assignment"))

    generation = await generate.run(PLAN, MIXED)

    assert generation.result.outcome is TurnOutcome.OUTBOUND_POLICY_BLOCKED
    assert generation.scan_rules == ("credential_assignment",)
    assert provider.requests == []


async def test_an_oversized_prompt_is_refused_before_the_model() -> None:
    generate, provider = generator(
        limits=AnswerLimits(
            max_prompt_bytes=500, max_output_tokens=3_000, temperature=0.3, reasoning_effort="low"
        )
    )

    with pytest.raises(PromptTooLargeError):
        await generate.run(PLAN, MIXED)
    assert provider.requests == []


def test_no_evidence_is_settled_without_the_model() -> None:
    result = settle_without_model(EvidenceSet())

    assert result is not None
    assert result.outcome is TurnOutcome.INSUFFICIENT_EVIDENCE


def test_restricted_only_evidence_is_settled_with_its_citations() -> None:
    only_private = EvidenceSet(restricted=(evidence(1, "機密一"), evidence(2, "機密二")))

    result = settle_without_model(only_private)

    assert result is not None
    assert result.outcome is TurnOutcome.EXTERNAL_GENERATION_BLOCKED
    assert result.restricted_citations == tuple(item.citation for item in only_private.restricted)
    assert result.answer is None


def test_sendable_evidence_needs_the_model() -> None:
    assert settle_without_model(MIXED) is None


async def test_unable_to_answer_has_no_answer() -> None:
    generate, _ = generator(answer_reply(status="unable_to_answer"))

    generation = await generate.run(PLAN, MIXED)

    assert generation.result.outcome is TurnOutcome.UNABLE_TO_ANSWER
    assert generation.result.answer is None


@pytest.mark.parametrize(
    "content",
    [
        answer_reply(("報名時間是十二月。", ["C3"])),
        answer_reply(("報名時間是十二月。", ["c1"])),
        answer_reply(("報名時間是十二月。", [])),
        answer_reply(("報名時間是十二月。", ["C1", "C1"])),
        answer_reply(("   ", ["C1"])),
        answer_reply(),
        answer_reply(("報名時間是十二月。", ["C1"]), status="unable_to_answer"),
        answer_reply(("報名時間是十二月。", ["C1"]), status="maybe"),
        json.dumps({"status": "answered", "answer_parts": [], "extra": 1}),
        json.dumps({"status": "answered", "answer_parts": [{"text": "x", "citation_ids": [1]}]}),
        json.dumps({"status": "answered", "answer_parts": [{"text": "x"}]}),
        json.dumps({"status": "answered", "answer_parts": "C1"}),
        json.dumps(["answered"]),
        "not json",
        answer_reply(("長" * 20_001, ["C1"])),
    ],
)
async def test_answers_outside_the_contract_are_rejected(content: str) -> None:
    generate, _ = generator(content)

    with pytest.raises(LlmContractError) as caught:
        await generate.run(PLAN, MIXED)
    assert caught.value.error_code == "invalid_answer"


async def test_an_unfinished_answer_is_rejected() -> None:
    truncated = LlmResponse(
        content=answer_reply(("報名時間是十二月。", ["C1"])), finish_reason="length", model="fake"
    )
    generate, _ = generator(truncated)

    with pytest.raises(LlmContractError):
        await generate.run(PLAN, MIXED)


async def test_the_conversation_history_reaches_the_scan_and_the_prompt() -> None:
    history = (
        HistoryMessage("user", "把黃小明的經歷整理給我"),
        HistoryMessage("assistant", "他有五年系統分析經驗。"),
    )
    scan = RecordingScan()
    generate, provider = generator(answer_reply(("重點一。", ["C1"])), scan=scan)

    await generate.run(PLAN, MIXED, history)

    [request] = provider.requests
    assert sent(request)["conversation_history"] == [
        {"role": "user", "text": "把黃小明的經歷整理給我"},
        {"role": "assistant", "text": "他有五年系統分析經驗。"},
    ]
    [scanned] = scan.scanned
    assert {"把黃小明的經歷整理給我", "他有五年系統分析經驗。"} <= set(scanned)


async def test_citation_labels_written_into_the_text_are_removed() -> None:
    generate, _ = generator(
        answer_reply(
            ("1. 五年經驗。 (C1)", ["C1"]),
            ("2. 熟悉 C# 與 Python（C1、C2）。", ["C1", "C2"]),
            ("[C2]", ["C2"]),
        )
    )

    generation = await generate.run(PLAN, MIXED)

    assert generation.result.answer is not None
    assert [part.text for part in generation.result.answer.parts] == [
        "1. 五年經驗。",
        "2. 熟悉 C# 與 Python。",
    ]
