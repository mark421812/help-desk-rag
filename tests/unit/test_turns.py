from __future__ import annotations

from collections.abc import Callable
from itertools import product
from uuid import uuid4

import pytest

from helpdesk.domain.answers import Answer, AnswerPart, AnswerStatus
from helpdesk.domain.conversations import (
    Conversation,
    ConversationId,
    ConversationStatus,
    Turn,
    TurnId,
    TurnOutcome,
    TurnResult,
    TurnState,
)
from helpdesk.domain.errors import (
    ArchivedConversationError,
    DomainValidationError,
    IdempotencyConflictError,
    InvalidTransitionError,
)
from helpdesk.domain.retrieval import QueryPlan
from tests.builders import FINGERPRINT, chunk_id, citation

PLAN = QueryPlan(
    original_question="那在職生呢？",
    standalone_question="115 碩專班在職生的報名資格是什麼？",
    language="zh-TW",
)

# The required state machine, written out independently of the implementation.
EXPECTED_MOVES: dict[TurnState, set[TurnState]] = {
    TurnState.ACCEPTED: {TurnState.RETRIEVING, TurnState.FAILED, TurnState.FAILED_UNKNOWN},
    TurnState.RETRIEVING: {
        TurnState.GENERATING,
        TurnState.COMPLETED,
        TurnState.FAILED,
        TurnState.FAILED_UNKNOWN,
    },
    TurnState.GENERATING: {TurnState.COMPLETED, TurnState.FAILED, TurnState.FAILED_UNKNOWN},
    TurnState.COMPLETED: set(),
    TurnState.FAILED: set(),
    TurnState.FAILED_UNKNOWN: set(),
}


def answered() -> TurnResult:
    cited = chunk_id()
    return TurnResult.from_answer(
        Answer(
            status=AnswerStatus.ANSWERED,
            parts=(AnswerPart(text="在職生需有兩年工作經驗。", chunk_ids=(cited,)),),
            citations=(citation(cited),),
        )
    )


def result_for(outcome: TurnOutcome) -> TurnResult:
    if outcome is TurnOutcome.ANSWERED:
        return answered()
    if outcome is TurnOutcome.EXTERNAL_GENERATION_BLOCKED:
        return TurnResult(outcome, restricted_citations=(citation(),))
    return TurnResult(outcome)


def natural_result(state: TurnState) -> TurnResult:
    if state is TurnState.GENERATING:
        return answered()
    return TurnResult(TurnOutcome.INSUFFICIENT_EVIDENCE)


MOVES: dict[TurnState, Callable[[Turn], Turn]] = {
    TurnState.RETRIEVING: lambda turn: turn.start_retrieving(PLAN),
    TurnState.GENERATING: lambda turn: turn.start_generating(),
    TurnState.COMPLETED: lambda turn: turn.complete(natural_result(turn.state)),
    TurnState.FAILED: lambda turn: turn.fail("provider_unavailable"),
    TurnState.FAILED_UNKNOWN: lambda turn: turn.fail(
        "processing_interrupted", outcome_unknown=True
    ),
}

PATHS: dict[TurnState, tuple[TurnState, ...]] = {
    TurnState.ACCEPTED: (),
    TurnState.RETRIEVING: (TurnState.RETRIEVING,),
    TurnState.GENERATING: (TurnState.RETRIEVING, TurnState.GENERATING),
    TurnState.COMPLETED: (TurnState.RETRIEVING, TurnState.COMPLETED),
    TurnState.FAILED: (TurnState.FAILED,),
    TurnState.FAILED_UNKNOWN: (TurnState.FAILED_UNKNOWN,),
}


def conversation(status: ConversationStatus = ConversationStatus.ACTIVE) -> Conversation:
    return Conversation(id=ConversationId(uuid4()), owner_id="anon-7f3a", status=status)


def accepted_turn() -> Turn:
    return Turn.accept(
        turn_id=TurnId(uuid4()),
        conversation=conversation(),
        idempotency_key=uuid4(),
        request_fingerprint=FINGERPRINT,
        question=PLAN.original_question,
        language="zh-TW",
    )


def turn_in(state: TurnState) -> Turn:
    turn = accepted_turn()
    for step in PATHS[state]:
        turn = MOVES[step](turn)
    assert turn.state is state
    return turn


@pytest.mark.parametrize(
    ("origin", "target"),
    [
        (origin, target)
        for origin, target in product(TurnState, MOVES)
        if target in EXPECTED_MOVES[origin]
    ],
)
def test_allowed_moves_reach_their_target(origin: TurnState, target: TurnState) -> None:
    assert MOVES[target](turn_in(origin)).state is target


@pytest.mark.parametrize(
    ("origin", "target"),
    [
        (origin, target)
        for origin, target in product(TurnState, MOVES)
        if target not in EXPECTED_MOVES[origin]
    ],
)
def test_moves_outside_the_table_are_rejected(origin: TurnState, target: TurnState) -> None:
    with pytest.raises(InvalidTransitionError):
        MOVES[target](turn_in(origin))


@pytest.mark.parametrize(
    ("origin", "outcome", "allowed"),
    [
        (TurnState.RETRIEVING, TurnOutcome.INSUFFICIENT_EVIDENCE, True),
        (TurnState.RETRIEVING, TurnOutcome.EXTERNAL_GENERATION_BLOCKED, True),
        (TurnState.RETRIEVING, TurnOutcome.ANSWERED, False),
        (TurnState.RETRIEVING, TurnOutcome.UNABLE_TO_ANSWER, False),
        (TurnState.RETRIEVING, TurnOutcome.OUTBOUND_POLICY_BLOCKED, False),
        (TurnState.GENERATING, TurnOutcome.ANSWERED, True),
        (TurnState.GENERATING, TurnOutcome.UNABLE_TO_ANSWER, True),
        (TurnState.GENERATING, TurnOutcome.OUTBOUND_POLICY_BLOCKED, True),
        (TurnState.GENERATING, TurnOutcome.INSUFFICIENT_EVIDENCE, False),
        (TurnState.GENERATING, TurnOutcome.EXTERNAL_GENERATION_BLOCKED, False),
    ],
)
def test_only_generation_produces_model_outcomes(
    origin: TurnState, outcome: TurnOutcome, allowed: bool
) -> None:
    turn = turn_in(origin)
    if allowed:
        completed = turn.complete(result_for(outcome))
        assert completed.state is TurnState.COMPLETED
        assert completed.result is not None
        assert completed.result.outcome is outcome
    else:
        with pytest.raises(InvalidTransitionError):
            turn.complete(result_for(outcome))


def test_follow_up_flow_records_the_standalone_question_and_the_answer() -> None:
    accepted = accepted_turn()
    result = answered()

    completed = accepted.start_retrieving(PLAN).start_generating().complete(result)

    assert accepted.state is TurnState.ACCEPTED
    assert accepted.standalone_question is None
    assert completed.original_question == PLAN.original_question
    assert completed.standalone_question == PLAN.standalone_question
    assert completed.result == result
    assert completed.error_code is None


def test_failures_keep_their_safe_error_code() -> None:
    failed = turn_in(TurnState.GENERATING).fail("rate_limited")

    assert failed.state is TurnState.FAILED
    assert failed.error_code == "rate_limited"
    assert failed.result is None


def test_error_codes_must_be_machine_codes() -> None:
    with pytest.raises(DomainValidationError):
        turn_in(TurnState.RETRIEVING).fail("Groq said: invalid key sk-123")


def test_archived_conversations_do_not_accept_turns() -> None:
    with pytest.raises(ArchivedConversationError):
        Turn.accept(
            turn_id=TurnId(uuid4()),
            conversation=conversation(ConversationStatus.ARCHIVED),
            idempotency_key=uuid4(),
            request_fingerprint=FINGERPRINT,
            question="報名截止日是哪天？",
        )


def test_a_reused_key_replays_the_same_request_and_conflicts_on_another() -> None:
    turn = accepted_turn()

    turn.ensure_same_request(FINGERPRINT)
    with pytest.raises(IdempotencyConflictError):
        turn.ensure_same_request("0" * 64)


@pytest.mark.parametrize(
    "fields",
    [
        {"state": TurnState.COMPLETED, "standalone_question": "q"},
        {"state": TurnState.FAILED},
        {"state": TurnState.RETRIEVING},
        {"state": TurnState.ACCEPTED, "error_code": "rate_limited"},
        {"state": TurnState.ACCEPTED, "standalone_question": "q"},
        {"request_fingerprint": "not-a-sha256"},
    ],
)
def test_inconsistent_stored_turns_are_rejected(fields: dict[str, object]) -> None:
    base: dict[str, object] = {
        "id": TurnId(uuid4()),
        "conversation_id": ConversationId(uuid4()),
        "idempotency_key": uuid4(),
        "request_fingerprint": FINGERPRINT,
        "original_question": "報名截止日是哪天？",
    }
    with pytest.raises(DomainValidationError):
        Turn(**(base | fields))  # type: ignore[arg-type]


def test_an_unable_answer_completes_without_answer_text() -> None:
    result = TurnResult.from_answer(Answer(status=AnswerStatus.UNABLE_TO_ANSWER))

    assert result.outcome is TurnOutcome.UNABLE_TO_ANSWER
    assert result.answer is None


@pytest.mark.parametrize(
    ("outcome", "fields"),
    [
        (TurnOutcome.ANSWERED, {}),
        (TurnOutcome.EXTERNAL_GENERATION_BLOCKED, {}),
        (TurnOutcome.INSUFFICIENT_EVIDENCE, {"restricted_citations": (citation(),)}),
    ],
)
def test_results_carry_exactly_what_their_outcome_shows(
    outcome: TurnOutcome, fields: dict[str, object]
) -> None:
    with pytest.raises(DomainValidationError):
        TurnResult(outcome, **fields)  # type: ignore[arg-type]


def test_a_rate_limited_failure_keeps_how_long_to_wait() -> None:
    failed = turn_in(TurnState.GENERATING).fail("provider_rate_limited", retry_after_seconds=17)

    assert (failed.state, failed.retry_after_seconds) == (TurnState.FAILED, 17)
    assert not failed.is_open
    assert turn_in(TurnState.GENERATING).is_open


@pytest.mark.parametrize(
    ("state", "unknown", "retry_after"),
    [
        (TurnState.GENERATING, True, 17),
        (TurnState.GENERATING, False, -1),
    ],
)
def test_only_definite_failures_carry_a_retry_after(
    state: TurnState, unknown: bool, retry_after: int
) -> None:
    with pytest.raises(DomainValidationError):
        turn_in(state).fail(
            "provider_rate_limited", outcome_unknown=unknown, retry_after_seconds=retry_after
        )


def test_a_retry_after_on_an_open_turn_is_rejected() -> None:
    with pytest.raises(DomainValidationError):
        Turn(
            id=TurnId(uuid4()),
            conversation_id=ConversationId(uuid4()),
            idempotency_key=uuid4(),
            request_fingerprint=FINGERPRINT,
            original_question="報名截止日是哪天？",
            retry_after_seconds=5,
        )
