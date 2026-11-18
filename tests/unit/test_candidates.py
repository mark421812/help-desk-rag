"""粗檢索：RRF 排序、關鍵字失敗退回向量、兩路走同一份權限複驗、被拒絕的不佔名次。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pytest

from helpdesk.application.retrieval.access import AccessPolicy, AclAccess, OpenAccess
from helpdesk.application.retrieval.candidates import CandidatePool, FindCandidates, merge_pools
from helpdesk.application.retrieval.ports import StoredChunk
from helpdesk.domain.access import AccessRule, PermissionEffect, Principal, PrincipalType
from helpdesk.domain.documents import ChunkId, DocumentStatus
from tests.retrieval_fakes import (
    TODAY,
    FakeCatalog,
    FakeKeywords,
    FakeVectors,
    FixedEmbedder,
    document,
    stored,
)

ALICE = Principal(user_id="alice", group_ids=("staff",))


async def find(
    vector: Sequence[StoredChunk],
    keyword: Sequence[StoredChunk] = (),
    *,
    access: AccessPolicy | None = None,
    keyword_error: Exception | None = None,
    question: str = "報名時間",
    limit: int = 24,
    extra: Sequence[StoredChunk] = (),
) -> CandidatePool:
    policy = access or OpenAccess()
    search = FindCandidates(
        embedder=FixedEmbedder(),
        vectors=FakeVectors(vector),
        keywords=FakeKeywords(keyword, keyword_error),
        catalog=FakeCatalog([*vector, *keyword, *extra]),
        keyword_min_score=3,
    )
    return await search.run(
        question,
        scope=policy.vector_scope(ALICE),
        admit=lambda candidate: policy.admits(candidate, ALICE, TODAY),
        limit=limit,
    )


def order(pool: CandidatePool) -> list[ChunkId]:
    return [candidate.chunk_id for candidate in pool.candidates]


async def test_rankings_are_fused_by_reciprocal_rank() -> None:
    a, b, c = stored(), stored(), stored()

    pool = await find([a, b, c], [c, a])

    assert order(pool) == [a.chunk.id, c.chunk.id, b.chunk.id]
    best = 2 / 61
    top = pool.candidates[0]
    assert top.fused_score == pytest.approx((1 / 61 + 1 / 62) / best)
    assert top.vector_score is not None and top.keyword_score is not None
    assert pool.candidates[2].keyword_score is None


async def test_ties_keep_the_vector_ranking_first() -> None:
    a, b = stored(), stored()

    pool = await find([a], [b])

    assert order(pool) == [a.chunk.id, b.chunk.id]
    assert pool.candidates[0].fused_score == pool.candidates[1].fused_score


async def test_the_fused_list_is_cut_to_the_limit() -> None:
    chunks = [stored() for _ in range(5)]

    pool = await find(chunks, limit=3)

    assert order(pool) == [candidate.chunk.id for candidate in chunks[:3]]


async def test_a_keyword_failure_falls_back_to_vector_results() -> None:
    a, b = stored(), stored()

    pool = await find([a, b], [b], keyword_error=RuntimeError("SELECT ... 報名時間"))

    assert order(pool) == [a.chunk.id, b.chunk.id]
    assert pool.keyword_failure == "RuntimeError"
    assert all(candidate.keyword_score is None for candidate in pool.candidates)
    assert pool.candidates[0].fused_score == pytest.approx(1.0)


async def test_a_question_without_keyword_terms_skips_keyword_search() -> None:
    a = stored()
    keywords = FakeKeywords([a])
    search = FindCandidates(
        embedder=FixedEmbedder(),
        vectors=FakeVectors([a]),
        keywords=keywords,
        catalog=FakeCatalog([a]),
        keyword_min_score=3,
    )

    pool = await search.run(
        "a", scope=OpenAccess().vector_scope(ALICE), admit=lambda _: True, limit=8
    )

    assert keywords.queries == []
    assert pool.keyword_failure is None
    assert order(pool) == [a.chunk.id]


def governance_cases() -> dict[str, StoredChunk]:
    deny = AccessRule(PrincipalType.USER, "alice", PermissionEffect.DENY)
    return {
        "public": stored(),
        "denied": stored(document(rules=(deny,))),
        "draft": stored(document(status=DocumentStatus.DRAFT)),
        "expired": stored(document(expires_at=date(2020, 1, 1))),
        "not_yet": stored(document(effective_at=date(2999, 1, 1))),
        "old_version": stored(current=False),
        "not_indexed": stored(indexed=False),
    }


@pytest.mark.parametrize(
    ("access", "expected"),
    [
        (OpenAccess(), {"public", "denied", "draft", "expired", "not_yet"}),
        (AclAccess(), {"public"}),
    ],
)
async def test_both_sides_pass_the_same_post_check(
    access: AccessPolicy, expected: set[str]
) -> None:
    cases = governance_cases()
    names = {stored.chunk.id: name for name, stored in cases.items()}

    vector_only = await find(list(cases.values()), access=access)
    keyword_only = await find([], list(cases.values()), access=access)

    assert {names[chunk_id] for chunk_id in order(vector_only)} == expected
    assert {names[chunk_id] for chunk_id in order(keyword_only)} == expected


async def test_rejected_chunks_do_not_use_up_the_pool() -> None:
    rejected = [stored(document(status=DocumentStatus.DRAFT)) for _ in range(9)]
    admitted = [stored() for _ in range(3)]

    pool = await find([*rejected, *admitted], access=AclAccess(), limit=1)

    assert order(pool) == [admitted[0].chunk.id]


async def test_the_standalone_question_is_what_gets_embedded_and_searched() -> None:
    embedder = FixedEmbedder()
    keywords = FakeKeywords()
    search = FindCandidates(
        embedder=embedder,
        vectors=FakeVectors(),
        keywords=keywords,
        catalog=FakeCatalog([]),
        keyword_min_score=3,
    )

    await search.run(
        "報名時間", scope=OpenAccess().vector_scope(ALICE), admit=lambda _: True, limit=8
    )

    assert embedder.queries == ["報名時間"]
    assert [term.text for term in keywords.queries[0].terms] == ["報名時間", "報名", "名時", "時間"]


async def test_merging_keeps_the_top_results_of_every_search_text() -> None:
    own_topic, shared, previous_topic = stored(), stored(), stored()
    question_alone = await find([own_topic, shared])
    with_history = await find([previous_topic, shared])

    merged = merge_pools([question_alone, with_history], limit=24)

    assert order(merged) == [shared.chunk.id, own_topic.chunk.id, previous_topic.chunk.id]
    assert merged.candidates[0].fused_score == pytest.approx((1 / 62 + 1 / 62) / (2 / 61))
    assert set(merged.chunks) == {own_topic.chunk.id, shared.chunk.id, previous_topic.chunk.id}


async def test_merging_one_search_text_changes_nothing() -> None:
    pool = await find([stored(), stored()])

    assert merge_pools([pool], limit=1) is pool


async def test_a_keyword_failure_in_any_search_text_is_reported() -> None:
    a = stored()
    healthy = await find([a])
    failed = await find([a], [a], keyword_error=RuntimeError("down"))

    assert merge_pools([healthy, failed], limit=8).keyword_failure == "RuntimeError"
