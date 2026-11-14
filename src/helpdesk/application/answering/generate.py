"""產生回答：外送掃描、prompt 預算、呼叫 LLM、嚴格解析與引用驗證。

只有 `EvidenceSet.sendable` 會進到外送掃描與 prompt，受限證據不離開本機。沒有可外送的證據時用不到
模型，由 `settle_without_model` 直接給出結果，回合在 retrieving 就結束。掃描器由 composition
root 依 `SENSITIVE_DATA_SCAN_ENABLED` 注入，這裡不判斷開關。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from helpdesk.application.answering.prompt import (
    GROUNDED_ANSWER_SCHEMA,
    answer_messages,
    citation_label,
)
from helpdesk.application.llm import (
    LlmContractError,
    LlmProvider,
    LlmRequest,
    LlmResponse,
    ReasoningEffort,
    TokenUsage,
    prompt_bytes,
)
from helpdesk.application.outbound_scan import OutboundScan
from helpdesk.domain.answers import Answer, AnswerPart, AnswerStatus
from helpdesk.domain.conversations import TurnOutcome, TurnResult
from helpdesk.domain.errors import DomainError, DomainValidationError
from helpdesk.domain.retrieval import Evidence, EvidenceSet, QueryPlan

MAX_ANSWER_CHARACTERS: Final = 20_000
INVALID_ANSWER: Final = "invalid_answer"


class PromptTooLargeError(DomainError):
    """prompt 超過 `MAX_PROMPT_TOKENS`。Settings 的啟動檢查保證滿額證據放得下，這是最後一道關。"""


@dataclass(frozen=True, slots=True)
class AnswerLimits:
    max_prompt_bytes: int
    max_output_tokens: int
    temperature: float
    reasoning_effort: ReasoningEffort


@dataclass(frozen=True, slots=True)
class Generation:
    """`scan_rules` 是外送掃描命中的規則代號，只在擋下時有值；`usage` 是 provider 回報的用量。
    兩者都不含內容，可以直接寫進 log。"""

    result: TurnResult
    scan_rules: tuple[str, ...] = ()
    usage: TokenUsage | None = None


def settle_without_model(evidence: EvidenceSet) -> TurnResult | None:
    """沒有可外送的證據時不呼叫模型；有就回 None，交給 `GenerateAnswer`。"""

    if evidence.sendable:
        return None
    if evidence.restricted:
        return TurnResult(
            TurnOutcome.EXTERNAL_GENERATION_BLOCKED,
            restricted_citations=tuple(item.citation for item in evidence.restricted),
        )
    return TurnResult(TurnOutcome.INSUFFICIENT_EVIDENCE)


class GenerateAnswer:
    def __init__(self, *, provider: LlmProvider, scan: OutboundScan, limits: AnswerLimits) -> None:
        self._provider = provider
        self._scan = scan
        self._limits = limits

    async def run(self, plan: QueryPlan, evidence: EvidenceSet) -> Generation:
        """回答 `plan.standalone_question`。前提是 `settle_without_model(evidence)` 為 None。"""

        sendable = evidence.sendable
        verdict = self._scan.scan(outbound_texts(plan, sendable))
        if not verdict.allowed:
            return Generation(
                TurnResult(TurnOutcome.OUTBOUND_POLICY_BLOCKED), scan_rules=verdict.rule_ids
            )
        messages = answer_messages(plan.standalone_question, plan.language, sendable)
        if prompt_bytes(messages) > self._limits.max_prompt_bytes:
            raise PromptTooLargeError("the answer prompt exceeds MAX_PROMPT_TOKENS")
        response = await self._provider.complete(
            LlmRequest(
                messages=messages,
                output_schema=GROUNDED_ANSWER_SCHEMA,
                max_output_tokens=self._limits.max_output_tokens,
                temperature=self._limits.temperature,
                reasoning_effort=self._limits.reasoning_effort,
            )
        )
        return Generation(
            TurnResult.from_answer(parse_answer(response, sendable)), usage=response.usage
        )


def outbound_texts(plan: QueryPlan, evidence: Sequence[Evidence]) -> list[str]:
    """即將送出的每一段文字：問題、語言，以及每段證據的內容、類型、標題與章節。"""

    texts = [plan.standalone_question, *([plan.language] if plan.language else [])]
    for item in evidence:
        texts.extend((item.content, item.document_type, item.citation.title))
        if item.citation.section_path is not None:
            texts.append(item.citation.section_path)
    return texts


def parse_answer(response: LlmResponse, evidence: Sequence[Evidence]) -> Answer:
    """嚴格解析模型輸出，把 C1..Cn 對回證據的 chunk。任何不合約定的輸出都是 contract error。

    引用清單依證據順序列出被引用到的證據；沒被任何一段引用的證據不列入。
    """

    if response.finish_reason != "stop":
        raise _invalid("the model did not finish its answer")
    if len(response.content) > MAX_ANSWER_CHARACTERS:
        raise _invalid("the answer exceeds the size limit")
    try:
        payload: Any = json.loads(response.content)
    except json.JSONDecodeError as exc:
        raise _invalid("the answer is not JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"status", "answer_parts"}:
        raise _invalid("the answer does not follow the schema")
    status = payload["status"]
    if status not in {member.value for member in AnswerStatus}:
        raise _invalid("the answer has an unknown status")
    raw_parts = payload["answer_parts"]
    if not isinstance(raw_parts, list):
        raise _invalid("answer_parts must be an array")
    by_label = {citation_label(position): item for position, item in enumerate(evidence, start=1)}
    parsed: list[tuple[str, list[str]]] = []
    for raw in raw_parts:
        if not isinstance(raw, dict) or set(raw) != {"text", "citation_ids"}:
            raise _invalid("an answer part does not follow the schema")
        text, labels = raw["text"], raw["citation_ids"]
        if not isinstance(text, str) or not isinstance(labels, list):
            raise _invalid("an answer part has invalid field types")
        if not all(isinstance(label, str) for label in labels):
            raise _invalid("citation ids must be strings")
        if any(label not in by_label for label in labels):
            raise _invalid("the answer cites an unknown citation id")
        parsed.append((text, labels))
    try:
        parts = tuple(
            AnswerPart(
                text=text, chunk_ids=tuple(by_label[label].citation.chunk_id for label in labels)
            )
            for text, labels in parsed
        )
        cited = {chunk_id for part in parts for chunk_id in part.chunk_ids}
        return Answer(
            status=AnswerStatus(status),
            parts=parts,
            citations=tuple(item.citation for item in evidence if item.citation.chunk_id in cited),
        )
    except DomainValidationError as exc:
        raise _invalid("the answer violates the grounded answer rules") from exc


def _invalid(message: str) -> LlmContractError:
    return LlmContractError(message, error_code=INVALID_ANSWER)
