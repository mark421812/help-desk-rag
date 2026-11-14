"""對話與回合 repository 在真的 PostgreSQL 上：擁有者範圍、keyset 分頁、Idempotency-Key 唯一、
compare-and-set 推進、歷史與回收的查詢，以及回答與引用的往返。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from helpdesk.application.conversation.ports import ConversationCursor
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
from helpdesk.domain.errors import ArchivedConversationError
from helpdesk.domain.retrieval import QueryPlan
from helpdesk.infrastructure.persistence.conversations import SqlConversations
from helpdesk.infrastructure.persistence.ingestion import session_factory
from tests.builders import FINGERPRINT, chunk_id, citation

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALICE, BOB = "browser:alice", "browser:bob"


@pytest.fixture(scope="module")
def database_url() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="asyncpg") as postgres:
        url = postgres.get_connection_url()
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("DATABASE_URL", url)
            patch.setenv("LLM_PROVIDER", "fake")
            command.upgrade(Config(str(PROJECT_ROOT / "alembic.ini")), "head")
        yield url


@pytest.fixture
async def engine(database_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(database_url)
    yield engine
    async with engine.begin() as connection:
        await connection.execute(text("TRUNCATE helpdesk.conversations CASCADE"))
    await engine.dispose()


@pytest.fixture
def store(engine: AsyncEngine) -> SqlConversations:
    return SqlConversations(session_factory(engine))


async def conversation(store: SqlConversations, owner: str = ALICE) -> Conversation:
    record = await store.add_conversation(
        Conversation(id=ConversationId(uuid4()), owner_id=owner, title="報名")
    )
    return record.conversation


def accepted(owner: Conversation, question: str = "報名時間？") -> Turn:
    return Turn.accept(
        turn_id=TurnId(uuid4()),
        conversation=owner,
        idempotency_key=uuid4(),
        request_fingerprint=FINGERPRINT,
        question=question,
        language="zh-TW",
    )


async def completed(store: SqlConversations, turn: Turn, result: TurnResult) -> Turn:
    plan = QueryPlan(turn.original_question, turn.original_question, turn.language)
    retrieving = turn.start_retrieving(plan)
    assert await store.advance(retrieving, previous=TurnState.ACCEPTED)
    generating = retrieving.start_generating()
    assert await store.advance(generating, previous=TurnState.RETRIEVING)
    done = generating.complete(result)
    assert await store.advance(done, previous=TurnState.GENERATING)
    return done


def answered() -> TurnResult:
    first, second = chunk_id(), chunk_id()
    return TurnResult.from_answer(
        Answer(
            status=AnswerStatus.ANSWERED,
            parts=(
                AnswerPart(text="十二月一日到十日。", chunk_ids=(first,)),
                AnswerPart(text="報名費一千五百元。", chunk_ids=(second, first)),
            ),
            citations=(citation(first), citation(second, title="收費標準")),
        )
    )


async def test_conversations_are_scoped_to_their_owner(store: SqlConversations) -> None:
    mine = await conversation(store)

    assert (await store.conversation(ALICE, mine.id)) is not None
    assert await store.conversation(BOB, mine.id) is None
    assert await store.save_conversation(Conversation(id=mine.id, owner_id=BOB, title="x")) is None
    assert await store.active_conversations(BOB, after=None, limit=10) == []


async def test_keyset_pages_follow_the_newest_activity(store: SqlConversations) -> None:
    first, second, third = [await conversation(store) for _ in range(3)]
    await conversation(store, BOB)
    assert await store.add_turn(accepted(first))
    archived = await conversation(store)
    await store.save_conversation(
        Conversation(id=archived.id, owner_id=ALICE, status=ConversationStatus.ARCHIVED)
    )

    page = await store.active_conversations(ALICE, after=None, limit=2)
    last = page[-1]
    rest = await store.active_conversations(
        ALICE, after=ConversationCursor(last.updated_at, last.conversation.id), limit=2
    )

    assert [record.conversation.id for record in page] == [first.id, third.id]
    assert [record.conversation.id for record in rest] == [second.id]


async def test_one_idempotency_key_yields_one_turn_even_under_a_race(
    store: SqlConversations, engine: AsyncEngine
) -> None:
    owner = await conversation(store)
    turn = accepted(owner)
    twin = Turn.accept(
        turn_id=TurnId(uuid4()),
        conversation=owner,
        idempotency_key=turn.idempotency_key,
        request_fingerprint=FINGERPRINT,
        question=turn.original_question,
    )

    results = await asyncio.gather(store.add_turn(turn), store.add_turn(twin))

    assert sorted(results) == [False, True]
    assert len(await store.turns(owner.id)) == 1
    winner = await store.turn_by_key(owner.id, turn.idempotency_key)
    assert winner is not None and winner.id in {turn.id, twin.id}
    with pytest.raises(IntegrityError):
        async with engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO helpdesk.turns (id, conversation_id, idempotency_key, "
                    "request_fingerprint, original_question, state) "
                    "VALUES (:id, :c, :k, :f, 'q', 'accepted')"
                ),
                {"id": uuid4(), "c": owner.id, "k": turn.idempotency_key, "f": FINGERPRINT},
            )


async def test_the_same_key_in_another_conversation_is_a_new_turn(store: SqlConversations) -> None:
    turn = accepted(await conversation(store))
    elsewhere = await conversation(store)
    copy = Turn.accept(
        turn_id=TurnId(uuid4()),
        conversation=elsewhere,
        idempotency_key=turn.idempotency_key,
        request_fingerprint=FINGERPRINT,
        question="報名時間？",
    )

    assert await store.add_turn(turn)
    assert await store.add_turn(copy)


async def test_archived_conversations_refuse_new_turns(store: SqlConversations) -> None:
    owner = await conversation(store)
    await store.save_conversation(
        Conversation(id=owner.id, owner_id=ALICE, status=ConversationStatus.ARCHIVED)
    )

    with pytest.raises(ArchivedConversationError):
        await store.add_turn(accepted(owner))


async def test_advancing_is_compare_and_set_and_results_round_trip(
    store: SqlConversations,
) -> None:
    owner = await conversation(store)
    turn = accepted(owner)
    await store.add_turn(turn)
    result = answered()

    done = await completed(store, turn, result)
    late = turn.fail("processing_interrupted", outcome_unknown=True)

    assert not await store.advance(late, previous=TurnState.ACCEPTED)
    assert await store.turn(owner.id, turn.id) == done
    assert await store.turn(ConversationId(uuid4()), turn.id) is None

    blocked = accepted(owner, "受限文件？")
    await store.add_turn(blocked)
    restricted = TurnResult(
        TurnOutcome.EXTERNAL_GENERATION_BLOCKED, restricted_citations=(citation(),)
    )
    plan = QueryPlan("受限文件？", "受限文件？")
    retrieving = blocked.start_retrieving(plan)
    await store.advance(retrieving, previous=TurnState.ACCEPTED)
    finished = retrieving.complete(restricted)
    await store.advance(finished, previous=TurnState.RETRIEVING)
    assert await store.turn(owner.id, blocked.id) == finished

    limited = accepted(owner, "被限流的問題？")
    await store.add_turn(limited)
    failed = limited.fail("provider_rate_limited", retry_after_seconds=17)
    await store.advance(failed, previous=TurnState.ACCEPTED)
    assert await store.turn(owner.id, limited.id) == failed


async def test_history_is_this_conversations_earlier_completed_turns(
    store: SqlConversations,
) -> None:
    owner, other = await conversation(store), await conversation(store)
    turns = [accepted(owner, f"問題 {index}") for index in range(4)]
    for turn in turns:
        await store.add_turn(turn)
    await store.add_turn(foreign := accepted(other, "別的對話"))
    for turn in (turns[0], turns[1], foreign):
        await completed(store, turn, answered())
    await store.advance(turns[2].fail("provider_unavailable"), previous=TurnState.ACCEPTED)

    history = await store.completed_before(turns[3], limit=3)
    latest = await store.completed_before(turns[3], limit=1)

    assert [turn.original_question for turn in history] == ["問題 0", "問題 1"]
    assert [turn.original_question for turn in latest] == ["問題 1"]
    assert [turn.id for turn in await store.turns(owner.id)] == [turn.id for turn in turns]


async def test_only_open_turns_without_progress_are_stale(
    store: SqlConversations, engine: AsyncEngine
) -> None:
    owner = await conversation(store)
    old, fresh, finished = accepted(owner, "舊的"), accepted(owner, "新的"), accepted(owner, "完成")
    for turn in (old, fresh, finished):
        await store.add_turn(turn)
    await completed(store, finished, answered())
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "UPDATE helpdesk.turns SET updated_at = now() - interval '10 minutes' "
                "WHERE id IN (:old, :finished)"
            ),
            {"old": old.id, "finished": finished.id},
        )

    stale = await store.stale_turns(older_than=timedelta(minutes=5))

    assert [turn.id for turn in stale] == [old.id]
