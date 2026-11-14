from __future__ import annotations

import pytest

from helpdesk.domain.answers import Answer, AnswerPart, AnswerStatus
from helpdesk.domain.errors import DomainValidationError
from helpdesk.domain.retrieval import Citation
from tests.builders import chunk_id, citation


def test_an_answer_lists_exactly_the_chunks_its_parts_cite() -> None:
    first, second = chunk_id(), chunk_id()

    answer = Answer(
        status=AnswerStatus.ANSWERED,
        parts=(
            AnswerPart(text="報名期間為 11 月 1 日至 11 月 20 日。", chunk_ids=(first,)),
            AnswerPart(text="需繳交兩年工作證明。", chunk_ids=(first, second)),
        ),
        citations=(citation(first), citation(second)),
    )

    assert [item.chunk_id for item in answer.citations] == [first, second]


def test_an_answered_result_needs_parts() -> None:
    with pytest.raises(DomainValidationError):
        Answer(status=AnswerStatus.ANSWERED, citations=(citation(),))


@pytest.mark.parametrize(
    ("parts", "citations"),
    [
        ((AnswerPart(text="無法回答", chunk_ids=(chunk_id(),)),), ()),
        ((), (citation(),)),
    ],
)
def test_unable_to_answer_carries_no_text_or_citations(
    parts: tuple[AnswerPart, ...], citations: tuple[Citation, ...]
) -> None:
    with pytest.raises(DomainValidationError):
        Answer(status=AnswerStatus.UNABLE_TO_ANSWER, parts=parts, citations=citations)


def test_a_part_citing_an_unlisted_chunk_is_rejected() -> None:
    listed = chunk_id()
    with pytest.raises(DomainValidationError):
        Answer(
            status=AnswerStatus.ANSWERED,
            parts=(AnswerPart(text="內容", chunk_ids=(listed, chunk_id())),),
            citations=(citation(listed),),
        )


def test_an_unused_citation_is_rejected() -> None:
    cited = chunk_id()
    with pytest.raises(DomainValidationError):
        Answer(
            status=AnswerStatus.ANSWERED,
            parts=(AnswerPart(text="內容", chunk_ids=(cited,)),),
            citations=(citation(cited), citation()),
        )


def test_the_same_chunk_cannot_be_listed_twice() -> None:
    cited = chunk_id()
    with pytest.raises(DomainValidationError):
        Answer(
            status=AnswerStatus.ANSWERED,
            parts=(AnswerPart(text="內容", chunk_ids=(cited,)),),
            citations=(citation(cited), citation(cited)),
        )


def test_every_part_must_cite_something() -> None:
    with pytest.raises(DomainValidationError):
        AnswerPart(text="沒有出處的句子", chunk_ids=())
