"""對話與回合的狀態機。

回合依 `_TRANSITIONS` 前進，轉移表之外的移動一律拒絕。追問改寫在 `accepted` 期間完成，
進入 `retrieving` 時記下獨立問題。終點狀態沒有出口，所以重送同一把 Idempotency-Key 時
只能回放結果，不會重跑。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Final, NewType
from uuid import UUID

from helpdesk.domain._checks import (
    MAX_LANGUAGE_LENGTH,
    MAX_PRINCIPAL_ID_LENGTH,
    MAX_QUESTION_LENGTH,
    optional_text,
    require_sha256,
    require_text,
    require_unique,
)
from helpdesk.domain.answers import Answer, AnswerStatus
from helpdesk.domain.errors import (
    ArchivedConversationError,
    DomainValidationError,
    IdempotencyConflictError,
    InvalidTransitionError,
)
from helpdesk.domain.retrieval import Citation, QueryPlan

ConversationId = NewType("ConversationId", UUID)
TurnId = NewType("TurnId", UUID)

MAX_CONVERSATION_TITLE_LENGTH: Final = 200

_ERROR_CODE: Final = re.compile(r"[a-z][a-z0-9_]{0,99}")


class ConversationStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


@dataclass(frozen=True, slots=True)
class Conversation:
    """一位擁有者的一串回合。封存後不再接受新回合。"""

    id: ConversationId
    owner_id: str
    title: str | None = None
    status: ConversationStatus = ConversationStatus.ACTIVE

    def __post_init__(self) -> None:
        require_text(self.owner_id, "owner_id", max_length=MAX_PRINCIPAL_ID_LENGTH)
        optional_text(self.title, "title", max_length=MAX_CONVERSATION_TITLE_LENGTH)


class TurnState(StrEnum):
    """回合在處理流程中的位置。

    `failed` 是確定失敗。`failed_unknown` 是處理中斷或模型結果無法確認，任何處理中的狀態
    都可能落到這裡；使用者只能換一把新的 key 重問。
    """

    ACCEPTED = "accepted"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    FAILED_UNKNOWN = "failed_unknown"


class TurnOutcome(StrEnum):
    ANSWERED = "answered"
    UNABLE_TO_ANSWER = "unable_to_answer"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    EXTERNAL_GENERATION_BLOCKED = "external_generation_blocked"
    OUTBOUND_POLICY_BLOCKED = "outbound_policy_blocked"


_TRANSITIONS: Final[Mapping[TurnState, frozenset[TurnState]]] = MappingProxyType(
    {
        TurnState.ACCEPTED: frozenset(
            {TurnState.RETRIEVING, TurnState.FAILED, TurnState.FAILED_UNKNOWN}
        ),
        TurnState.RETRIEVING: frozenset(
            {
                TurnState.GENERATING,
                TurnState.COMPLETED,
                TurnState.FAILED,
                TurnState.FAILED_UNKNOWN,
            }
        ),
        TurnState.GENERATING: frozenset(
            {TurnState.COMPLETED, TurnState.FAILED, TurnState.FAILED_UNKNOWN}
        ),
        TurnState.COMPLETED: frozenset(),
        TurnState.FAILED: frozenset(),
        TurnState.FAILED_UNKNOWN: frozenset(),
    }
)

# 沒有可外送證據時檢索後就結束；凡是看過模型或外送掃描的結果，只能來自 generating。
_COMPLETIONS: Final[Mapping[TurnState, frozenset[TurnOutcome]]] = MappingProxyType(
    {
        TurnState.RETRIEVING: frozenset(
            {TurnOutcome.INSUFFICIENT_EVIDENCE, TurnOutcome.EXTERNAL_GENERATION_BLOCKED}
        ),
        TurnState.GENERATING: frozenset(
            {
                TurnOutcome.ANSWERED,
                TurnOutcome.UNABLE_TO_ANSWER,
                TurnOutcome.OUTBOUND_POLICY_BLOCKED,
            }
        ),
    }
)

_FAILED: Final = frozenset({TurnState.FAILED, TurnState.FAILED_UNKNOWN})
# 還在處理中的狀態，也就是轉移表裡還有出口的狀態。
OPEN_TURN_STATES: Final = frozenset(state for state, moves in _TRANSITIONS.items() if moves)
_PLANNED: Final = frozenset({TurnState.RETRIEVING, TurnState.GENERATING, TurnState.COMPLETED})


@dataclass(frozen=True, slots=True)
class TurnResult:
    """完成的回合給使用者看的結果。

    只有 `answered` 帶回答。`external_generation_blocked` 帶不能外送的文件引用，讓使用者
    知道可以自己去讀哪些文件。其他結果只有 outcome。
    """

    outcome: TurnOutcome
    answer: Answer | None = None
    restricted_citations: tuple[Citation, ...] = ()

    def __post_init__(self) -> None:
        if (self.outcome is TurnOutcome.ANSWERED) != (self.answer is not None):
            raise DomainValidationError("exactly the answered results carry an answer")
        if self.answer is not None and self.answer.status is not AnswerStatus.ANSWERED:
            raise DomainValidationError("an answered result needs an answered answer")
        if (self.outcome is TurnOutcome.EXTERNAL_GENERATION_BLOCKED) != bool(
            self.restricted_citations
        ):
            raise DomainValidationError(
                "exactly the external_generation_blocked results list restricted citations"
            )
        require_unique(
            [citation.chunk_id for citation in self.restricted_citations], "restricted_citations"
        )

    @classmethod
    def from_answer(cls, answer: Answer) -> TurnResult:
        if answer.status is AnswerStatus.ANSWERED:
            return cls(TurnOutcome.ANSWERED, answer=answer)
        return cls(TurnOutcome.UNABLE_TO_ANSWER)


@dataclass(frozen=True, slots=True)
class Turn:
    """對話中的一問一答，以及它在狀態機裡的位置。

    同一個對話裡 `idempotency_key` 唯一。重送同一把 key 時，`request_fingerprint` 相同就
    回放既有回合，不同就是衝突。provider 限流而確定失敗的回合帶 `retry_after_seconds`，告訴
    使用者多久後可以再問。
    """

    id: TurnId
    conversation_id: ConversationId
    idempotency_key: UUID
    request_fingerprint: str
    original_question: str
    language: str | None = None
    standalone_question: str | None = None
    state: TurnState = TurnState.ACCEPTED
    result: TurnResult | None = None
    error_code: str | None = None
    retry_after_seconds: int | None = None

    def __post_init__(self) -> None:
        require_sha256(self.request_fingerprint, "request_fingerprint")
        require_text(self.original_question, "original_question", max_length=MAX_QUESTION_LENGTH)
        optional_text(self.language, "language", max_length=MAX_LANGUAGE_LENGTH)
        optional_text(
            self.standalone_question, "standalone_question", max_length=MAX_QUESTION_LENGTH
        )
        if self.error_code is not None and not _ERROR_CODE.fullmatch(self.error_code):
            raise DomainValidationError("error_code must be a lowercase machine code")
        if (self.state is TurnState.COMPLETED) != (self.result is not None):
            raise DomainValidationError("exactly the completed turns carry a result")
        if (self.state in _FAILED) != (self.error_code is not None):
            raise DomainValidationError("exactly the failed turns carry an error code")
        if self.retry_after_seconds is not None and (
            self.state is not TurnState.FAILED or self.retry_after_seconds < 0
        ):
            raise DomainValidationError("only failed turns carry a non-negative retry-after")
        if self.state is TurnState.ACCEPTED and self.standalone_question is not None:
            raise DomainValidationError("an accepted turn has no standalone question yet")
        if self.state in _PLANNED and self.standalone_question is None:
            raise DomainValidationError("a turn past understanding needs its standalone question")

    @classmethod
    def accept(
        cls,
        *,
        turn_id: TurnId,
        conversation: Conversation,
        idempotency_key: UUID,
        request_fingerprint: str,
        question: str,
        language: str | None = None,
    ) -> Turn:
        if conversation.status is ConversationStatus.ARCHIVED:
            raise ArchivedConversationError("archived conversations do not accept new turns")
        return cls(
            id=turn_id,
            conversation_id=conversation.id,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            original_question=question,
            language=language,
        )

    def start_retrieving(self, plan: QueryPlan) -> Turn:
        self._require_move(TurnState.RETRIEVING)
        return replace(
            self, state=TurnState.RETRIEVING, standalone_question=plan.standalone_question
        )

    def start_generating(self) -> Turn:
        self._require_move(TurnState.GENERATING)
        return replace(self, state=TurnState.GENERATING)

    def complete(self, result: TurnResult) -> Turn:
        if result.outcome not in _COMPLETIONS.get(self.state, frozenset()):
            raise InvalidTransitionError(f"a {self.state} turn cannot complete as {result.outcome}")
        self._require_move(TurnState.COMPLETED)
        return replace(self, state=TurnState.COMPLETED, result=result)

    def fail(
        self,
        error_code: str,
        *,
        outcome_unknown: bool = False,
        retry_after_seconds: int | None = None,
    ) -> Turn:
        target = TurnState.FAILED_UNKNOWN if outcome_unknown else TurnState.FAILED
        self._require_move(target)
        return replace(
            self, state=target, error_code=error_code, retry_after_seconds=retry_after_seconds
        )

    @property
    def is_open(self) -> bool:
        return self.state in OPEN_TURN_STATES

    def ensure_same_request(self, request_fingerprint: str) -> None:
        if request_fingerprint != self.request_fingerprint:
            raise IdempotencyConflictError("the idempotency key was reused for another request")

    def _require_move(self, target: TurnState) -> None:
        if target not in _TRANSITIONS[self.state]:
            raise InvalidTransitionError(f"a {self.state} turn cannot move to {target}")
