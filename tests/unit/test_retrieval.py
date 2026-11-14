from __future__ import annotations

import pytest

from helpdesk.domain.errors import DomainValidationError
from helpdesk.domain.retrieval import Candidate, Evidence, EvidenceSet, QueryPlan, RetrievalStatus
from tests.builders import chunk_id, citation


def evidence(rank: int) -> Evidence:
    return Evidence(
        rank=rank,
        citation=citation(),
        content=f"第 {rank} 段內容",
        document_type="policy",
        token_count=40,
        score=1 / rank,
    )


@pytest.mark.parametrize(
    ("sendable", "restricted", "status"),
    [
        ((evidence(1),), (evidence(2),), RetrievalStatus.EVIDENCE_FOUND),
        ((), (evidence(1),), RetrievalStatus.EXTERNAL_GENERATION_BLOCKED),
        ((), (), RetrievalStatus.INSUFFICIENT_EVIDENCE),
    ],
)
def test_status_follows_which_pile_has_evidence(
    sendable: tuple[Evidence, ...], restricted: tuple[Evidence, ...], status: RetrievalStatus
) -> None:
    assert EvidenceSet(sendable=sendable, restricted=restricted).status is status


def test_ordered_merges_both_piles_by_rank() -> None:
    first, second, third, fourth = (evidence(rank) for rank in range(1, 5))

    evidence_set = EvidenceSet(sendable=(first, fourth), restricted=(second, third))

    assert evidence_set.ordered == (first, second, third, fourth)


@pytest.mark.parametrize(
    ("sendable", "restricted"),
    [
        ((evidence(1), evidence(3)), ()),
        ((evidence(1),), (evidence(1),)),
        ((evidence(2), evidence(1)), ()),
        ((evidence(2),), ()),
    ],
)
def test_ranks_must_be_one_contiguous_ascending_sequence(
    sendable: tuple[Evidence, ...], restricted: tuple[Evidence, ...]
) -> None:
    with pytest.raises(DomainValidationError):
        EvidenceSet(sendable=sendable, restricted=restricted)


def test_the_same_chunk_cannot_be_evidence_twice() -> None:
    first = evidence(1)
    duplicate = Evidence(
        rank=2,
        citation=first.citation,
        content=first.content,
        document_type=first.document_type,
        token_count=first.token_count,
        score=first.score,
    )
    with pytest.raises(DomainValidationError):
        EvidenceSet(sendable=(first,), restricted=(duplicate,))


def test_a_candidate_comes_from_at_least_one_search() -> None:
    with pytest.raises(DomainValidationError):
        Candidate(chunk_id=chunk_id(), fused_score=0.5)

    keyword_only = Candidate(chunk_id=chunk_id(), fused_score=0.5, keyword_score=7.0)
    assert keyword_only.vector_score is None


def test_a_query_plan_needs_a_standalone_question() -> None:
    with pytest.raises(DomainValidationError):
        QueryPlan(original_question="那在職生呢？", standalone_question="  ")
