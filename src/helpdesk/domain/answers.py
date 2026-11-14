"""通過引用驗證的回答。

Domain 只用 chunk ID 表示引用。模型在 prompt 裡看到的引用代號由 answering 模組自行對照
回 chunk ID，不進到 domain。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from helpdesk.domain._checks import require_text, require_unique
from helpdesk.domain.documents import ChunkId
from helpdesk.domain.errors import DomainValidationError
from helpdesk.domain.retrieval import Citation


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    UNABLE_TO_ANSWER = "unable_to_answer"


@dataclass(frozen=True, slots=True)
class AnswerPart:
    """回答的一段，至少引用一段證據。"""

    text: str
    chunk_ids: tuple[ChunkId, ...]

    def __post_init__(self) -> None:
        require_text(self.text, "answer part text")
        if not self.chunk_ids:
            raise DomainValidationError("an answer part must cite at least one chunk")
        require_unique(self.chunk_ids, "answer part chunk_ids")


@dataclass(frozen=True, slots=True)
class Answer:
    """有回答時每段都有引用，且引用清單剛好是各段引用的聯集；無法回答時兩者都空。"""

    status: AnswerStatus
    parts: tuple[AnswerPart, ...] = ()
    citations: tuple[Citation, ...] = ()

    def __post_init__(self) -> None:
        if self.status is AnswerStatus.UNABLE_TO_ANSWER:
            if self.parts or self.citations:
                raise DomainValidationError("unable_to_answer must not carry parts or citations")
            return
        if not self.parts:
            raise DomainValidationError("an answered result must contain answer parts")
        listed = [citation.chunk_id for citation in self.citations]
        require_unique(listed, "citations")
        cited = {chunk_id for part in self.parts for chunk_id in part.chunk_ids}
        if cited != set(listed):
            raise DomainValidationError("citations must be exactly the chunks the parts cite")
