"""提問流程：接受回合、在背景依回合狀態機處理、回收中斷的回合。

- 接受：同一個對話裡同一把 Idempotency-Key 只會有一個回合。指紋相同就回放既有回合，不同就是
  衝突。指紋只看正規化後的問題與語言。
- 處理：accepted 期間讀這個對話已完成的回合當歷史並理解問題（追問會改寫）→ retrieving 記下
  獨立問題並檢索 → 沒有可外送證據就在這裡完成 → generating 呼叫模型 → completed。每一步各用一
  個短 transaction，而且只在資料庫裡的回合還停在上一個狀態時才寫入；回收程序已經把它標成
  `failed_unknown` 的話，處理就此停下，不會覆蓋。
- 失敗：依例外型別對照成機器代碼。provider 逾時或處理中斷時，無法確認模型是否已經產生結果，
  所以是 `failed_unknown`；其他是確定的 `failed`。兩者都只能換一把新的 key 重問。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
from typing import Final
from uuid import UUID, uuid4

from helpdesk.application.answering.generate import (
    GenerateAnswer,
    PromptTooLargeError,
    settle_without_model,
)
from helpdesk.application.conversation.ports import (
    ConversationNotFoundError,
    ConversationRepository,
)
from helpdesk.application.llm import (
    LlmContractError,
    LlmError,
    LlmRateLimitedError,
    LlmTimeoutError,
    TokenUsage,
)
from helpdesk.application.retrieval.ports import QueryTooLongError
from helpdesk.application.retrieval.retrieve import RetrieveEvidence
from helpdesk.application.retrieval.understanding import (
    HistoryMessage,
    RewriteOutcome,
    UnderstandQuestion,
    normalize_question,
)
from helpdesk.domain.access import Principal
from helpdesk.domain.conversations import ConversationId, Turn, TurnId
from helpdesk.domain.errors import IdempotencyConflictError

MAX_RETRY_AFTER_SECONDS: Final = 86_400


@dataclass(frozen=True, slots=True)
class Failure:
    code: str
    unknown: bool = False
    retry_after_seconds: int | None = None


INTERRUPTED: Final = Failure("processing_interrupted", unknown=True)


def request_fingerprint(question: str, language: str | None) -> str:
    """正規化後的問題與語言的 canonical JSON 的 SHA-256。"""

    payload = json.dumps(
        {"language": language, "question": question},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def history_messages(turns: Sequence[Turn]) -> list[HistoryMessage]:
    """提問一律放進歷史；助理的文字只取有回答的回合，其他結果沒有可供改寫的內容。"""

    messages: list[HistoryMessage] = []
    for turn in turns:
        messages.append(HistoryMessage(role="user", text=turn.original_question))
        answer = turn.result.answer if turn.result is not None else None
        if answer is not None:
            text = "\n".join(part.text for part in answer.parts)
            messages.append(HistoryMessage(role="assistant", text=text))
    return messages


def retrieval_failure(error: Exception) -> Failure:
    if isinstance(error, QueryTooLongError):
        return Failure("question_too_long")
    return Failure("retrieval_failed")


def generation_failure(error: Exception) -> Failure:
    if isinstance(error, LlmRateLimitedError):
        retry_after = error.retry_after_seconds
        return Failure(
            "provider_rate_limited",
            retry_after_seconds=(
                None if retry_after is None else min(max(retry_after, 0), MAX_RETRY_AFTER_SECONDS)
            ),
        )
    if isinstance(error, LlmTimeoutError | TimeoutError):
        return Failure("provider_timeout", unknown=True)
    if isinstance(error, LlmContractError):
        return Failure("provider_contract_error")
    if isinstance(error, LlmError):
        return Failure("provider_unavailable")
    if isinstance(error, PromptTooLargeError):
        return Failure("prompt_too_large")
    return INTERRUPTED


@dataclass(frozen=True, slots=True)
class Submission:
    """`created` 為 False 表示這是既有回合的回放。"""

    turn: Turn
    created: bool


class SubmitTurn:
    def __init__(
        self, repository: ConversationRepository, *, new_id: Callable[[], UUID] = uuid4
    ) -> None:
        self._repository = repository
        self._new_id = new_id

    async def run(
        self,
        owner_id: str,
        conversation_id: ConversationId,
        *,
        idempotency_key: UUID,
        question: str,
        language: str | None,
    ) -> Submission:
        processed = normalize_question(question)
        fingerprint = request_fingerprint(processed, language)
        record = await self._repository.conversation(owner_id, conversation_id)
        if record is None:
            raise ConversationNotFoundError("the conversation does not exist")
        existing = await self._repository.turn_by_key(conversation_id, idempotency_key)
        if existing is None:
            turn = Turn.accept(
                turn_id=TurnId(self._new_id()),
                conversation=record.conversation,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                question=processed,
                language=language,
            )
            if await self._repository.add_turn(turn):
                return Submission(turn, created=True)
            existing = await self._repository.turn_by_key(conversation_id, idempotency_key)
            if existing is None:
                raise IdempotencyConflictError("the idempotency key is in use by another request")
        existing.ensure_same_request(fingerprint)
        return Submission(existing, created=False)


@dataclass(slots=True)
class TurnReport:
    """一個回合處理完的摘要。只有狀態、代碼與數量，可以直接寫進 log。

    `stopped` 表示資料庫裡的回合已經被別人推進（例如回收程序），這次處理就此停下。
    `error_type` 是造成失敗的例外型別或 provider 錯誤代碼。
    """

    turn: Turn
    stopped: bool = False
    rewrite: RewriteOutcome | None = None
    rewrite_detail: str | None = None
    sendable: int = 0
    restricted: int = 0
    keyword_failure: str | None = None
    scan_rules: tuple[str, ...] = ()
    rewrite_usage: TokenUsage | None = None
    answer_usage: TokenUsage | None = None
    error_type: str | None = None


class ProcessTurn:
    def __init__(
        self,
        *,
        repository: ConversationRepository,
        understand: UnderstandQuestion,
        retrieve: RetrieveEvidence,
        generate: GenerateAnswer,
        history_turns: int,
    ) -> None:
        self._repository = repository
        self._understand = understand
        self._retrieve = retrieve
        self._generate = generate
        self._history_turns = history_turns

    async def run(self, turn: Turn, principal: Principal) -> TurnReport:
        report = TurnReport(turn)
        try:
            await self._steps(report, principal)
        except asyncio.CancelledError:
            with suppress(Exception):
                await self._fail(report, INTERRUPTED)
            raise
        except Exception as error:
            report.error_type = type(error).__name__
            with suppress(Exception):
                await self._fail(report, INTERRUPTED)
        return report

    async def _steps(self, report: TurnReport, principal: Principal) -> None:
        turn = report.turn
        earlier = await self._repository.completed_before(turn, limit=self._history_turns)
        understanding = await self._understand.run(
            turn.original_question, turn.language, history_messages(earlier)
        )
        report.rewrite, report.rewrite_detail = understanding.rewrite, understanding.detail
        report.rewrite_usage = understanding.usage
        if not await self._move(report, turn.start_retrieving(understanding.plan)):
            return
        try:
            retrieval = await self._retrieve.search(understanding, principal)
        except Exception as error:
            report.error_type = type(error).__name__
            await self._fail(report, retrieval_failure(error))
            return
        evidence = retrieval.evidence
        report.sendable, report.restricted = len(evidence.sendable), len(evidence.restricted)
        report.keyword_failure = retrieval.keyword_failure
        settled = settle_without_model(evidence)
        if settled is not None:
            await self._move(report, report.turn.complete(settled))
            return
        if not await self._move(report, report.turn.start_generating()):
            return
        try:
            generation = await self._generate.run(
                understanding.plan, evidence, understanding.history
            )
        except Exception as error:
            code = error.error_code if isinstance(error, LlmError) else None
            report.error_type = code or type(error).__name__
            await self._fail(report, generation_failure(error))
            return
        report.scan_rules = generation.scan_rules
        report.answer_usage = generation.usage
        await self._move(report, report.turn.complete(generation.result))

    async def _move(self, report: TurnReport, moved: Turn) -> bool:
        if not await self._repository.advance(moved, previous=report.turn.state):
            report.stopped = True
            return False
        report.turn = moved
        return True

    async def _fail(self, report: TurnReport, failure: Failure) -> None:
        if report.turn.is_open and not report.stopped:
            await self._move(
                report,
                report.turn.fail(
                    failure.code,
                    outcome_unknown=failure.unknown,
                    retry_after_seconds=failure.retry_after_seconds,
                ),
            )


class RecoverStaleTurns:
    """把超過門檻沒有進展的處理中回合標成 `failed_unknown`。重跑只會處理還沒被標記的回合。"""

    def __init__(self, repository: ConversationRepository, *, stale_after: timedelta) -> None:
        self._repository = repository
        self._stale_after = stale_after

    async def run(self) -> int:
        recovered = 0
        for turn in await self._repository.stale_turns(older_than=self._stale_after):
            failed = turn.fail(INTERRUPTED.code, outcome_unknown=True)
            if await self._repository.advance(failed, previous=turn.state):
                recovered += 1
        return recovered
