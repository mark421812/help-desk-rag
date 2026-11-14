"""Retrieval 各階段之間傳遞的資料。

理解問題產生 `QueryPlan`，粗檢索產生 `Candidate`，權限過濾後組裝成 `EvidenceSet`。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from helpdesk.domain._checks import (
    MAX_LANGUAGE_LENGTH,
    MAX_QUESTION_LENGTH,
    optional_positive,
    optional_text,
    require_finite,
    require_positive,
    require_text,
    require_unique,
)
from helpdesk.domain.documents import (
    MAX_DOCUMENT_TYPE_LENGTH,
    MAX_TITLE_LENGTH,
    ChunkId,
    DocumentId,
    VersionId,
)
from helpdesk.domain.errors import DomainValidationError

_MAX_CONTENT_LENGTH: Final = 200_000


class RetrievalStatus(StrEnum):
    EVIDENCE_FOUND = "evidence_found"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    EXTERNAL_GENERATION_BLOCKED = "external_generation_blocked"


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """理解問題的結果。

    追問會被改寫成不依賴前文的 `standalone_question`；不需要改寫或改寫失敗時，它就等於
    `original_question`。檢索只用 `standalone_question`。
    """

    original_question: str
    standalone_question: str
    language: str | None = None

    def __post_init__(self) -> None:
        require_text(self.original_question, "original_question", max_length=MAX_QUESTION_LENGTH)
        require_text(
            self.standalone_question, "standalone_question", max_length=MAX_QUESTION_LENGTH
        )
        optional_text(self.language, "language", max_length=MAX_LANGUAGE_LENGTH)


@dataclass(frozen=True, slots=True)
class Candidate:
    """粗檢索的一個候選 chunk。

    向量與關鍵字兩路至少有一路命中；沒命中的那一路分數為 None。之後的排序只看兩路排名
    以 RRF 合併後的 `fused_score`。
    """

    chunk_id: ChunkId
    fused_score: float
    vector_score: float | None = None
    keyword_score: float | None = None

    def __post_init__(self) -> None:
        if self.vector_score is None and self.keyword_score is None:
            raise DomainValidationError("a candidate needs a vector or keyword score")
        for name, score in (
            ("vector_score", self.vector_score),
            ("keyword_score", self.keyword_score),
        ):
            if score is not None:
                require_finite(score, name)
        require_finite(self.fused_score, "fused_score")
        if self.fused_score <= 0:
            raise DomainValidationError("fused_score must be positive")


@dataclass(frozen=True, slots=True)
class Citation:
    """可以回給使用者看的引用。型別刻意沒有檔案路徑或 URL，想塞也塞不進來。"""

    chunk_id: ChunkId
    document_id: DocumentId
    version_id: VersionId
    title: str
    section_path: str | None = None
    page_number: int | None = None

    def __post_init__(self) -> None:
        require_text(self.title, "citation title", max_length=MAX_TITLE_LENGTH)
        optional_text(self.section_path, "citation section_path")
        optional_positive(self.page_number, "citation page_number")


@dataclass(frozen=True, slots=True)
class Evidence:
    """通過權限過濾、選進證據集的一段內容。`rank` 從 1 開始，越小越相關。"""

    rank: int
    citation: Citation
    content: str
    document_type: str
    token_count: int
    score: float

    def __post_init__(self) -> None:
        require_positive(self.rank, "rank")
        require_text(self.content, "evidence content", max_length=_MAX_CONTENT_LENGTH)
        require_text(self.document_type, "document_type", max_length=MAX_DOCUMENT_TYPE_LENGTH)
        require_positive(self.token_count, "token_count")
        require_finite(self.score, "score")


@dataclass(frozen=True, slots=True)
class EvidenceSet:
    """排序好的證據，依能否送到外部 LLM 分成兩堆。

    一段證據放在哪一堆，就是它能不能外送的唯一依據，所以 `Evidence` 本身沒有外送旗標，
    也就不會出現「放在可外送那堆卻標記不可外送」的狀態。rank 在兩堆之間是全域連續的
    1..n，`ordered` 依 rank 合併回單一序列。
    """

    sendable: tuple[Evidence, ...] = ()
    restricted: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        for name, pile in (("sendable", self.sendable), ("restricted", self.restricted)):
            ranks = [evidence.rank for evidence in pile]
            if ranks != sorted(ranks):
                raise DomainValidationError(f"{name} evidence must be in ascending rank order")
        ordered = self.ordered
        if [evidence.rank for evidence in ordered] != list(range(1, len(ordered) + 1)):
            raise DomainValidationError("evidence ranks must be unique and contiguous from 1")
        require_unique([evidence.citation.chunk_id for evidence in ordered], "evidence chunks")

    @property
    def ordered(self) -> tuple[Evidence, ...]:
        return tuple(sorted(self.sendable + self.restricted, key=lambda evidence: evidence.rank))

    @property
    def status(self) -> RetrievalStatus:
        if self.sendable:
            return RetrievalStatus.EVIDENCE_FOUND
        if self.restricted:
            return RetrievalStatus.EXTERNAL_GENERATION_BLOCKED
        return RetrievalStatus.INSUFFICIENT_EVIDENCE
