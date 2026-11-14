"""對話與回合的 PostgreSQL repository。

每個方法各自開一個短 transaction。新增回合時先 `FOR UPDATE` 鎖住對話，同一個對話的提問因此
依序寫入，`created_at` 用 `clock_timestamp()` 記下實際寫入的先後。推進回合用 compare-and-set：
只更新仍停在上一個狀態的列，回收程序與處理流程不會互相覆蓋。
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import Interval, func, insert, literal, select, tuple_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from helpdesk.application.conversation.ports import (
    ConversationCursor,
    ConversationNotFoundError,
    ConversationRecord,
)
from helpdesk.domain.answers import Answer, AnswerPart, AnswerStatus
from helpdesk.domain.conversations import (
    OPEN_TURN_STATES,
    Conversation,
    ConversationId,
    ConversationStatus,
    Turn,
    TurnId,
    TurnOutcome,
    TurnResult,
    TurnState,
)
from helpdesk.domain.documents import ChunkId, DocumentId, VersionId
from helpdesk.domain.errors import ArchivedConversationError
from helpdesk.domain.retrieval import Citation
from helpdesk.infrastructure.persistence.models import ConversationModel, TurnModel


class SqlConversations:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def add_conversation(self, conversation: Conversation) -> ConversationRecord:
        async with self._sessions() as session, session.begin():
            row = (
                await session.execute(
                    insert(ConversationModel)
                    .values(
                        id=conversation.id,
                        owner_id=conversation.owner_id,
                        title=conversation.title,
                        status=conversation.status.value,
                    )
                    .returning(ConversationModel.created_at, ConversationModel.updated_at)
                )
            ).one()
        return ConversationRecord(conversation, created_at=row[0], updated_at=row[1])

    async def conversation(
        self, owner_id: str, conversation_id: ConversationId
    ) -> ConversationRecord | None:
        async with self._sessions() as session:
            row = await session.scalar(
                select(ConversationModel).where(
                    ConversationModel.id == conversation_id, ConversationModel.owner_id == owner_id
                )
            )
        return None if row is None else _record(row)

    async def active_conversations(
        self, owner_id: str, *, after: ConversationCursor | None, limit: int
    ) -> list[ConversationRecord]:
        statement = select(ConversationModel).where(
            ConversationModel.owner_id == owner_id,
            ConversationModel.status == ConversationStatus.ACTIVE.value,
        )
        if after is not None:
            statement = statement.where(
                tuple_(ConversationModel.updated_at, ConversationModel.id)
                < tuple_(literal(after.updated_at), literal(after.conversation_id))
            )
        statement = statement.order_by(
            ConversationModel.updated_at.desc(), ConversationModel.id.desc()
        ).limit(limit)
        async with self._sessions() as session:
            rows = await session.scalars(statement)
            return [_record(row) for row in rows]

    async def save_conversation(self, conversation: Conversation) -> ConversationRecord | None:
        async with self._sessions() as session, session.begin():
            row = (
                await session.execute(
                    update(ConversationModel)
                    .where(
                        ConversationModel.id == conversation.id,
                        ConversationModel.owner_id == conversation.owner_id,
                    )
                    .values(
                        title=conversation.title,
                        status=conversation.status.value,
                        updated_at=func.now(),
                    )
                    .returning(ConversationModel.created_at, ConversationModel.updated_at)
                )
            ).one_or_none()
        if row is None:
            return None
        return ConversationRecord(conversation, created_at=row[0], updated_at=row[1])

    async def add_turn(self, turn: Turn) -> bool:
        async with self._sessions() as session, session.begin():
            status = await session.scalar(
                select(ConversationModel.status)
                .where(ConversationModel.id == turn.conversation_id)
                .with_for_update()
            )
            if status is None:
                raise ConversationNotFoundError("the conversation does not exist")
            if status != ConversationStatus.ACTIVE.value:
                raise ArchivedConversationError("archived conversations do not accept new turns")
            inserted = await session.scalar(
                pg_insert(TurnModel)
                .values(
                    **_turn_values(turn),
                    created_at=func.clock_timestamp(),
                    updated_at=func.clock_timestamp(),
                )
                .on_conflict_do_nothing(constraint="uq_turns_conversation_idempotency_key")
                .returning(TurnModel.id)
            )
            if inserted is None:
                return False
            await session.execute(
                update(ConversationModel)
                .where(ConversationModel.id == turn.conversation_id)
                .values(updated_at=func.now())
            )
        return True

    async def turn(self, conversation_id: ConversationId, turn_id: TurnId) -> Turn | None:
        async with self._sessions() as session:
            row = await session.scalar(
                select(TurnModel).where(
                    TurnModel.conversation_id == conversation_id, TurnModel.id == turn_id
                )
            )
        return None if row is None else _turn(row)

    async def turn_by_key(
        self, conversation_id: ConversationId, idempotency_key: UUID
    ) -> Turn | None:
        async with self._sessions() as session:
            row = await session.scalar(
                select(TurnModel).where(
                    TurnModel.conversation_id == conversation_id,
                    TurnModel.idempotency_key == idempotency_key,
                )
            )
        return None if row is None else _turn(row)

    async def turns(self, conversation_id: ConversationId) -> list[Turn]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(TurnModel)
                .where(TurnModel.conversation_id == conversation_id)
                .order_by(TurnModel.created_at, TurnModel.id)
            )
            return [_turn(row) for row in rows]

    async def completed_before(self, turn: Turn, *, limit: int) -> list[Turn]:
        anchor = select(TurnModel.created_at).where(TurnModel.id == turn.id).scalar_subquery()
        async with self._sessions() as session:
            rows = list(
                await session.scalars(
                    select(TurnModel)
                    .where(
                        TurnModel.conversation_id == turn.conversation_id,
                        TurnModel.state == TurnState.COMPLETED.value,
                        TurnModel.created_at < anchor,
                    )
                    .order_by(TurnModel.created_at.desc())
                    .limit(limit)
                )
            )
        return [_turn(row) for row in reversed(rows)]

    async def advance(self, turn: Turn, *, previous: TurnState) -> bool:
        async with self._sessions() as session, session.begin():
            moved = await session.scalar(
                update(TurnModel)
                .where(TurnModel.id == turn.id, TurnModel.state == previous.value)
                .values(
                    standalone_question=turn.standalone_question,
                    state=turn.state.value,
                    error_code=turn.error_code,
                    retry_after_seconds=turn.retry_after_seconds,
                    updated_at=func.now(),
                    completed_at=None if turn.is_open else func.now(),
                    **_result_values(turn.result),
                )
                .returning(TurnModel.id)
            )
        return moved is not None

    async def stale_turns(self, *, older_than: timedelta) -> list[Turn]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(TurnModel).where(
                    TurnModel.state.in_([state.value for state in OPEN_TURN_STATES]),
                    TurnModel.updated_at < func.now() - literal(older_than, Interval()),
                )
            )
            return [_turn(row) for row in rows]


def _record(row: ConversationModel) -> ConversationRecord:
    return ConversationRecord(
        conversation=Conversation(
            id=ConversationId(row.id),
            owner_id=row.owner_id,
            title=row.title,
            status=ConversationStatus(row.status),
        ),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _turn_values(turn: Turn) -> dict[str, Any]:
    return {
        "id": turn.id,
        "conversation_id": turn.conversation_id,
        "idempotency_key": turn.idempotency_key,
        "request_fingerprint": turn.request_fingerprint,
        "original_question": turn.original_question,
        "language": turn.language,
        "standalone_question": turn.standalone_question,
        "state": turn.state.value,
        "error_code": turn.error_code,
        "retry_after_seconds": turn.retry_after_seconds,
        **_result_values(turn.result),
    }


def _result_values(result: TurnResult | None) -> dict[str, Any]:
    if result is None:
        return {"outcome": None, "answer": None, "citations": []}
    answer = result.answer
    citations = answer.citations if answer is not None else result.restricted_citations
    return {
        "outcome": result.outcome.value,
        "answer": (
            None
            if answer is None
            else [
                {"text": part.text, "chunk_ids": [str(chunk_id) for chunk_id in part.chunk_ids]}
                for part in answer.parts
            ]
        ),
        "citations": [_citation_values(citation) for citation in citations],
    }


def _citation_values(citation: Citation) -> dict[str, Any]:
    return {
        "chunk_id": str(citation.chunk_id),
        "document_id": str(citation.document_id),
        "version_id": str(citation.version_id),
        "title": citation.title,
        "section_path": citation.section_path,
        "page_number": citation.page_number,
    }


def _turn(row: TurnModel) -> Turn:
    return Turn(
        id=TurnId(row.id),
        conversation_id=ConversationId(row.conversation_id),
        idempotency_key=row.idempotency_key,
        request_fingerprint=row.request_fingerprint,
        original_question=row.original_question,
        language=row.language,
        standalone_question=row.standalone_question,
        state=TurnState(row.state),
        result=_result(row),
        error_code=row.error_code,
        retry_after_seconds=row.retry_after_seconds,
    )


def _result(row: TurnModel) -> TurnResult | None:
    if row.outcome is None:
        return None
    outcome = TurnOutcome(row.outcome)
    citations = tuple(_citation(item) for item in row.citations)
    if outcome is TurnOutcome.ANSWERED:
        parts = tuple(
            AnswerPart(
                text=part["text"],
                chunk_ids=tuple(ChunkId(UUID(chunk_id)) for chunk_id in part["chunk_ids"]),
            )
            for part in row.answer or []
        )
        return TurnResult.from_answer(Answer(AnswerStatus.ANSWERED, parts, citations))
    if outcome is TurnOutcome.EXTERNAL_GENERATION_BLOCKED:
        return TurnResult(outcome, restricted_citations=citations)
    return TurnResult(outcome)


def _citation(item: dict[str, Any]) -> Citation:
    return Citation(
        chunk_id=ChunkId(UUID(item["chunk_id"])),
        document_id=DocumentId(UUID(item["document_id"])),
        version_id=VersionId(UUID(item["version_id"])),
        title=item["title"],
        section_path=item["section_path"],
        page_number=item["page_number"],
    )
