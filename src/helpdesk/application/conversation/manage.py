"""對話管理：建立、列出、改名、封存，以及讀取回合。每個操作都只看擁有者自己的對話。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from uuid import UUID, uuid4

from helpdesk.application.conversation.ports import (
    ConversationCursor,
    ConversationNotFoundError,
    ConversationRecord,
    ConversationRepository,
    TurnNotFoundError,
)
from helpdesk.domain.conversations import (
    Conversation,
    ConversationId,
    ConversationStatus,
    Turn,
    TurnId,
)


@dataclass(frozen=True, slots=True)
class ConversationPage:
    records: tuple[ConversationRecord, ...]
    next_cursor: ConversationCursor | None


class ManageConversations:
    def __init__(
        self, repository: ConversationRepository, *, new_id: Callable[[], UUID] = uuid4
    ) -> None:
        self._repository = repository
        self._new_id = new_id

    async def create(self, owner_id: str, title: str | None) -> ConversationRecord:
        conversation = Conversation(
            id=ConversationId(self._new_id()), owner_id=owner_id, title=title
        )
        return await self._repository.add_conversation(conversation)

    async def page(
        self, owner_id: str, *, after: ConversationCursor | None, limit: int
    ) -> ConversationPage:
        found = await self._repository.active_conversations(owner_id, after=after, limit=limit + 1)
        records = tuple(found[:limit])
        last = records[-1] if len(found) > limit else None
        return ConversationPage(
            records=records,
            next_cursor=(
                None if last is None else ConversationCursor(last.updated_at, last.conversation.id)
            ),
        )

    async def get(self, owner_id: str, conversation_id: ConversationId) -> ConversationRecord:
        record = await self._repository.conversation(owner_id, conversation_id)
        if record is None:
            raise ConversationNotFoundError("the conversation does not exist")
        return record

    async def update(
        self,
        owner_id: str,
        conversation_id: ConversationId,
        *,
        title: str | None = None,
        archive: bool = False,
    ) -> ConversationRecord:
        """`title` 為 None 表示不改名；封存之後不再出現在列表，也不再接受提問。"""

        conversation = (await self.get(owner_id, conversation_id)).conversation
        if title is not None:
            conversation = replace(conversation, title=title)
        if archive:
            conversation = replace(conversation, status=ConversationStatus.ARCHIVED)
        saved = await self._repository.save_conversation(conversation)
        if saved is None:
            raise ConversationNotFoundError("the conversation does not exist")
        return saved

    async def turns(self, owner_id: str, conversation_id: ConversationId) -> list[Turn]:
        await self.get(owner_id, conversation_id)
        return await self._repository.turns(conversation_id)

    async def turn(self, owner_id: str, conversation_id: ConversationId, turn_id: TurnId) -> Turn:
        await self.get(owner_id, conversation_id)
        turn = await self._repository.turn(conversation_id, turn_id)
        if turn is None:
            raise TurnNotFoundError("the turn does not exist")
        return turn
