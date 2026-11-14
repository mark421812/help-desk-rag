"""對話與回合的儲存 port，以及列表用的讀取資料。

對話的讀寫都帶擁有者；找不到與不屬於這個擁有者的對話一律回 None，呼叫端因此分不出兩者，
對外都是 404。回合只經由已確認擁有者的對話 ID 讀取。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol
from uuid import UUID

from helpdesk.domain.conversations import (
    Conversation,
    ConversationId,
    Turn,
    TurnId,
    TurnState,
)
from helpdesk.domain.errors import DomainError


class ConversationNotFoundError(DomainError):
    """對話不存在，或不屬於這個擁有者。"""


class TurnNotFoundError(DomainError):
    """這個對話裡沒有這個回合。"""


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    """`updated_at` 是最近一次活動：建立、改名、封存或收到新的提問。"""

    conversation: Conversation
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ConversationCursor:
    """keyset 分頁的位置：列表依 `(updated_at, id)` 由新到舊，下一頁從這一筆之後開始。"""

    updated_at: datetime
    conversation_id: ConversationId


class ConversationRepository(Protocol):
    async def add_conversation(self, conversation: Conversation) -> ConversationRecord: ...

    async def conversation(
        self, owner_id: str, conversation_id: ConversationId
    ) -> ConversationRecord | None: ...

    async def active_conversations(
        self, owner_id: str, *, after: ConversationCursor | None, limit: int
    ) -> list[ConversationRecord]:
        """未封存的對話，最近有活動的在前。"""
        ...

    async def save_conversation(self, conversation: Conversation) -> ConversationRecord | None:
        """寫回標題與狀態並更新活動時間；不屬於 `conversation.owner_id` 時回 None。"""
        ...

    async def add_turn(self, turn: Turn) -> bool:
        """鎖住對話後新增回合並更新活動時間。

        同一把 Idempotency-Key 已經有回合時不新增、回 False。對話在這期間被封存時丟
        `ArchivedConversationError`。
        """
        ...

    async def turn(self, conversation_id: ConversationId, turn_id: TurnId) -> Turn | None: ...

    async def turn_by_key(
        self, conversation_id: ConversationId, idempotency_key: UUID
    ) -> Turn | None: ...

    async def turns(self, conversation_id: ConversationId) -> list[Turn]:
        """依提問先後排列。"""
        ...

    async def completed_before(self, turn: Turn, *, limit: int) -> list[Turn]:
        """同一個對話裡比 `turn` 早、已完成的最後 `limit` 個回合，依先後排列。"""
        ...

    async def advance(self, turn: Turn, *, previous: TurnState) -> bool:
        """只在資料庫裡的回合仍是 `previous` 時寫入 `turn`，否則回 False、不寫入。"""
        ...

    async def stale_turns(self, *, older_than: timedelta) -> list[Turn]:
        """處理中、而且超過 `older_than` 沒有任何狀態變化的回合。"""
        ...
