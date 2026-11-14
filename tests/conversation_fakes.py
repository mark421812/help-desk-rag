"""對話與提問流程的記憶體 adapter，以及把整條提問流程組起來的測試用 pipeline。"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from fastapi.testclient import TestClient

from helpdesk.application.answering.generate import AnswerLimits, GenerateAnswer
from helpdesk.application.conversation.ask import ProcessTurn, RecoverStaleTurns, SubmitTurn
from helpdesk.application.conversation.manage import ManageConversations
from helpdesk.application.conversation.ports import ConversationCursor, ConversationRecord
from helpdesk.application.identity import IdentityResolver
from helpdesk.application.outbound_scan import OutboundScan
from helpdesk.application.retrieval.access import AccessPolicy, OpenAccess
from helpdesk.application.retrieval.assembly import AssemblyLimits
from helpdesk.application.retrieval.candidates import FindCandidates
from helpdesk.application.retrieval.ports import StoredChunk
from helpdesk.application.retrieval.retrieve import RetrieveEvidence
from helpdesk.application.retrieval.understanding import RewriteLimits, UnderstandQuestion
from helpdesk.config import Settings
from helpdesk.domain.conversations import (
    Conversation,
    ConversationId,
    ConversationStatus,
    Turn,
    TurnId,
    TurnState,
)
from helpdesk.domain.errors import ArchivedConversationError
from helpdesk.infrastructure.identity import AnonymousIdentity
from helpdesk.infrastructure.llm.fake import FakeLlmProvider, Reply
from helpdesk.presentation.api.app import create_app
from helpdesk.presentation.api.background import TurnRunner
from helpdesk.presentation.api.composition import ApiRuntime
from tests.retrieval_fakes import (
    TODAY,
    FakeCatalog,
    FakeKeywords,
    FakeVectors,
    FixedEmbedder,
    RecordingScan,
)

STALE_AFTER = timedelta(minutes=5)


class InMemoryConversations:
    """記憶體版的 `ConversationRepository`。

    `moves` 依序記下每一次寫入的回合狀態（新增時的 accepted 也算）。時間只在寫入時往前走一秒，
    測試可以用 `now` 往前撥，模擬回合停滯。
    """

    def __init__(self) -> None:
        self.now = datetime(2026, 11, 13, 9, 0, tzinfo=UTC)
        self.moves: list[tuple[TurnId, TurnState]] = []
        self._conversations: dict[ConversationId, ConversationRecord] = {}
        self._turns: dict[TurnId, Turn] = {}
        self._touched: dict[TurnId, datetime] = {}

    def _tick(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now

    async def add_conversation(self, conversation: Conversation) -> ConversationRecord:
        at = self._tick()
        record = ConversationRecord(conversation, created_at=at, updated_at=at)
        self._conversations[conversation.id] = record
        return record

    async def conversation(
        self, owner_id: str, conversation_id: ConversationId
    ) -> ConversationRecord | None:
        record = self._conversations.get(conversation_id)
        if record is None or record.conversation.owner_id != owner_id:
            return None
        return record

    async def active_conversations(
        self, owner_id: str, *, after: ConversationCursor | None, limit: int
    ) -> list[ConversationRecord]:
        rows = sorted(
            (
                record
                for record in self._conversations.values()
                if record.conversation.owner_id == owner_id
                and record.conversation.status is ConversationStatus.ACTIVE
            ),
            key=lambda record: (record.updated_at, record.conversation.id),
            reverse=True,
        )
        if after is not None:
            position = (after.updated_at, after.conversation_id)
            rows = [row for row in rows if (row.updated_at, row.conversation.id) < position]
        return rows[:limit]

    async def save_conversation(self, conversation: Conversation) -> ConversationRecord | None:
        record = await self.conversation(conversation.owner_id, conversation.id)
        if record is None:
            return None
        saved = ConversationRecord(conversation, record.created_at, self._tick())
        self._conversations[conversation.id] = saved
        return saved

    async def add_turn(self, turn: Turn) -> bool:
        record = self._conversations[turn.conversation_id]
        if record.conversation.status is ConversationStatus.ARCHIVED:
            raise ArchivedConversationError("archived")
        if await self.turn_by_key(turn.conversation_id, turn.idempotency_key) is not None:
            return False
        at = self._tick()
        self._turns[turn.id] = turn
        self._touched[turn.id] = at
        self.moves.append((turn.id, turn.state))
        self._conversations[turn.conversation_id] = ConversationRecord(
            record.conversation, record.created_at, at
        )
        return True

    async def turn(self, conversation_id: ConversationId, turn_id: TurnId) -> Turn | None:
        turn = self._turns.get(turn_id)
        return turn if turn is not None and turn.conversation_id == conversation_id else None

    async def turn_by_key(
        self, conversation_id: ConversationId, idempotency_key: UUID
    ) -> Turn | None:
        return next(
            (
                turn
                for turn in self._turns.values()
                if turn.conversation_id == conversation_id
                and turn.idempotency_key == idempotency_key
            ),
            None,
        )

    async def turns(self, conversation_id: ConversationId) -> list[Turn]:
        return [turn for turn in self._turns.values() if turn.conversation_id == conversation_id]

    async def completed_before(self, turn: Turn, *, limit: int) -> list[Turn]:
        earlier: list[Turn] = []
        for other in self._turns.values():
            if other.id == turn.id:
                break
            if other.conversation_id == turn.conversation_id and other.state is TurnState.COMPLETED:
                earlier.append(other)
        return earlier[-limit:]

    async def advance(self, turn: Turn, *, previous: TurnState) -> bool:
        if self._turns[turn.id].state is not previous:
            return False
        self._turns[turn.id] = turn
        self._touched[turn.id] = self._tick()
        self.moves.append((turn.id, turn.state))
        return True

    async def stale_turns(self, *, older_than: timedelta) -> list[Turn]:
        return [
            turn
            for turn in self._turns.values()
            if turn.is_open and self._touched[turn.id] < self.now - older_than
        ]

    def states_of(self, turn_id: TurnId) -> list[TurnState]:
        return [state for moved, state in self.moves if moved == turn_id]


def answer_reply(*parts: tuple[str, Sequence[str]], status: str = "answered") -> str:
    return json.dumps(
        {
            "status": status,
            "answer_parts": [{"text": text, "citation_ids": list(ids)} for text, ids in parts],
        },
        ensure_ascii=False,
    )


def rewrite_reply(question: str) -> str:
    return json.dumps({"standalone_question": question}, ensure_ascii=False)


@dataclass
class Pipeline:
    repository: InMemoryConversations
    provider: FakeLlmProvider
    vectors: FakeVectors
    retrieve: RetrieveEvidence
    conversations: ManageConversations
    submit: SubmitTurn
    process: ProcessTurn
    recover: RecoverStaleTurns


def pipeline(
    chunks: Sequence[StoredChunk],
    *,
    replies: Sequence[Reply] = (),
    access: AccessPolicy | None = None,
    scan: OutboundScan | None = None,
    keyword_error: Exception | None = None,
    max_prompt_bytes: int = 48_000,
    provider: FakeLlmProvider | None = None,
) -> Pipeline:
    repository = InMemoryConversations()
    provider = provider or FakeLlmProvider(replies)
    vectors = FakeVectors(chunks)
    outbound = scan or RecordingScan()
    understand = UnderstandQuestion(
        provider=provider,
        scan=outbound,
        limits=RewriteLimits(
            history_turns=3,
            turn_max_chars=500,
            max_chars=500,
            timeout_seconds=1,
            max_prompt_bytes=max_prompt_bytes,
            reasoning_effort="low",
        ),
    )
    retrieve = RetrieveEvidence(
        understand=understand,
        candidates=FindCandidates(
            embedder=FixedEmbedder(),
            vectors=vectors,
            keywords=FakeKeywords(chunks, keyword_error),
            catalog=FakeCatalog(chunks),
            keyword_min_score=3,
        ),
        access=access or OpenAccess(),
        limits=AssemblyLimits(max_evidence=8, max_chunks_per_document=6, max_context_tokens=6_000),
        candidate_count=24,
        today=lambda: TODAY,
    )
    return Pipeline(
        repository=repository,
        provider=provider,
        vectors=vectors,
        retrieve=retrieve,
        conversations=ManageConversations(repository),
        submit=SubmitTurn(repository),
        process=ProcessTurn(
            repository=repository,
            understand=understand,
            retrieve=retrieve,
            generate=GenerateAnswer(
                provider=provider,
                scan=outbound,
                limits=AnswerLimits(
                    max_prompt_bytes=max_prompt_bytes,
                    max_output_tokens=3_000,
                    temperature=0.3,
                    reasoning_effort="low",
                ),
            ),
            history_turns=3,
        ),
        recover=RecoverStaleTurns(repository, stale_after=STALE_AFTER),
    )


async def yes() -> bool:
    return True


async def nothing() -> None:
    return None


def api_runtime(
    pipe: Pipeline,
    *,
    identity: IdentityResolver | None = None,
    readiness: dict[str, Callable[[], Awaitable[bool]]] | None = None,
) -> ApiRuntime:
    turns = TurnRunner(pipe.process, pipe.recover, max_concurrency=2)

    async def close() -> None:
        await turns.close()

    return ApiRuntime(
        retrieve=pipe.retrieve,
        conversations=pipe.conversations,
        submit=pipe.submit,
        turns=turns,
        identity=identity or AnonymousIdentity(),
        readiness=readiness or {"postgres": yes, "qdrant": yes, "embedding_model": yes},
        warm_up=lambda: None,
        close=close,
    )


@contextmanager
def client(api: ApiRuntime, **settings: Any) -> Iterator[TestClient]:
    app = create_app(
        Settings(_env_file=None, llm_provider="fake", **settings), runtime_factory=lambda _: api
    )
    with TestClient(app) as test_client:
        yield test_client
