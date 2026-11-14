"""組裝證據：每份文件上限、證據數上限、只算可外送的 token 預算、依外送類別去重與分堆。"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from helpdesk.application.retrieval.assembly import AssemblyLimits, assemble
from helpdesk.application.retrieval.ports import StoredChunk
from helpdesk.domain.documents import Document
from helpdesk.domain.retrieval import Candidate, EvidenceSet, RetrievalStatus
from tests.retrieval_fakes import document, stored

LIMITS = AssemblyLimits(max_evidence=8, max_chunks_per_document=6, max_context_tokens=6_000)


def everything(_: Document) -> bool:
    return True


def run(
    chunks: Sequence[StoredChunk],
    *,
    sendable: Callable[[Document], bool] = everything,
    limits: AssemblyLimits = LIMITS,
) -> EvidenceSet:
    candidates = [
        Candidate(chunk_id=item.chunk.id, fused_score=1.0 - index / 100, vector_score=0.5)
        for index, item in enumerate(chunks)
    ]
    return assemble(candidates, {item.chunk.id: item for item in chunks}, sendable, limits)


def ids(evidence: EvidenceSet) -> list[object]:
    return [item.citation.chunk_id for item in evidence.ordered]


def test_each_document_contributes_at_most_its_cap() -> None:
    brochure, resume = document(), document(title="履歷")
    many = [stored(brochure) for _ in range(8)]
    others = [stored(resume) for _ in range(2)]

    evidence = run([*many, *others])

    assert ids(evidence) == [item.chunk.id for item in [*many[:6], *others]]
    assert [item.rank for item in evidence.ordered] == list(range(1, 9))


def test_selection_stops_at_the_evidence_limit() -> None:
    chunks = [stored(document()) for _ in range(12)]

    assert len(run(chunks).ordered) == 8


def test_the_token_budget_counts_sendable_evidence_and_skips_what_does_not_fit() -> None:
    owner = document()
    big, too_big, small = (
        stored(owner, tokens=60),
        stored(owner, tokens=50),
        stored(owner, tokens=30),
    )

    evidence = run(
        [big, too_big, small],
        limits=AssemblyLimits(max_evidence=8, max_chunks_per_document=6, max_context_tokens=100),
    )

    assert ids(evidence) == [big.chunk.id, small.chunk.id]


def test_restricted_evidence_does_not_spend_the_budget() -> None:
    private = document(external_llm_allowed=False)
    restricted = stored(private, tokens=500)
    allowed = stored(tokens=90)

    evidence = run(
        [restricted, allowed],
        sendable=lambda owner: owner.metadata.external_llm_allowed,
        limits=AssemblyLimits(max_evidence=8, max_chunks_per_document=6, max_context_tokens=100),
    )

    assert [item.citation.chunk_id for item in evidence.restricted] == [restricted.chunk.id]
    assert [item.citation.chunk_id for item in evidence.sendable] == [allowed.chunk.id]
    assert evidence.status is RetrievalStatus.EVIDENCE_FOUND


def test_duplicate_content_is_kept_once_per_governance_class() -> None:
    private = document(external_llm_allowed=False)
    first = stored(private, content="同一段內容")
    same_class = stored(private, content="同一段內容")
    other_class = stored(content="同一段內容")

    evidence = run(
        [first, same_class, other_class],
        sendable=lambda owner: owner.metadata.external_llm_allowed,
    )

    assert ids(evidence) == [first.chunk.id, other_class.chunk.id]


def test_only_restricted_evidence_blocks_external_generation() -> None:
    private = document(external_llm_allowed=False)

    evidence = run([stored(private)], sendable=lambda owner: owner.metadata.external_llm_allowed)

    assert evidence.sendable == ()
    assert evidence.status is RetrievalStatus.EXTERNAL_GENERATION_BLOCKED
    assert run([]).status is RetrievalStatus.INSUFFICIENT_EVIDENCE


def test_citations_carry_title_section_and_page() -> None:
    item = stored(document(title="115 碩專簡章"), page=10)

    [evidence] = run([item]).ordered

    assert (
        evidence.citation.title,
        evidence.citation.section_path,
        evidence.citation.page_number,
    ) == (
        "115 碩專簡章",
        "報名",
        10,
    )
    assert evidence.content == item.chunk.content
