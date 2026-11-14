"""retrieval 的 use case：理解問題、粗檢索、權限過濾、組裝證據。

`AccessPolicy` 由 composition root 依 `ACCESS_CONTROL_ENABLED` 選定後注入，這裡不判斷開關。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

from helpdesk.application.retrieval.access import AccessPolicy
from helpdesk.application.retrieval.assembly import AssemblyLimits, assemble
from helpdesk.application.retrieval.candidates import FindCandidates
from helpdesk.application.retrieval.ports import StoredChunk
from helpdesk.application.retrieval.understanding import (
    HistoryMessage,
    Understanding,
    UnderstandQuestion,
)
from helpdesk.domain.access import Principal
from helpdesk.domain.retrieval import Candidate, EvidenceSet


@dataclass(frozen=True, slots=True)
class Retrieval:
    understanding: Understanding
    evidence: EvidenceSet
    candidates: tuple[Candidate, ...]
    keyword_failure: str | None = None


class RetrieveEvidence:
    def __init__(
        self,
        *,
        understand: UnderstandQuestion,
        candidates: FindCandidates,
        access: AccessPolicy,
        limits: AssemblyLimits,
        candidate_count: int,
        today: Callable[[], date],
    ) -> None:
        self._understand = understand
        self._candidates = candidates
        self._access = access
        self._limits = limits
        self._candidate_count = candidate_count
        self._today = today

    async def run(
        self,
        question: str,
        *,
        language: str | None,
        history: Sequence[HistoryMessage],
        principal: Principal,
    ) -> Retrieval:
        understanding = await self._understand.run(question, language, history)
        return await self.search(understanding, principal)

    async def search(self, understanding: Understanding, principal: Principal) -> Retrieval:
        """用已經理解好的問題檢索；提問流程在兩步之間記下回合的狀態。"""

        today = self._today()

        def admit(stored: StoredChunk) -> bool:
            return self._access.admits(stored, principal, today)

        pool = await self._candidates.run(
            understanding.plan.standalone_question,
            scope=self._access.vector_scope(principal),
            admit=admit,
            limit=self._candidate_count,
        )
        evidence = assemble(pool.candidates, pool.chunks, self._access.sendable, self._limits)
        return Retrieval(
            understanding=understanding,
            evidence=evidence,
            candidates=pool.candidates,
            keyword_failure=pool.keyword_failure,
        )
