"""關鍵字詞與權重。"""

from __future__ import annotations

from helpdesk.application.retrieval.keywords import keyword_query


def terms(question: str) -> list[str]:
    return [term.text for term in keyword_query(question, 3).terms]


def test_a_cjk_run_yields_the_whole_run_and_its_bigrams() -> None:
    assert terms("報名日期") == ["報名日期", "報名", "名日", "日期"]


def test_ascii_codes_and_names_are_kept_whole_and_lowercased() -> None:
    found = terms("請問 HR-2024 與 Alice")

    assert "hr-2024" in found
    assert "alice" in found


def test_single_characters_are_dropped_and_terms_are_capped() -> None:
    assert terms("a 我 b") == []
    assert len(terms("".join(chr(0x4E00 + index) for index in range(200)))) == 32


def test_longer_terms_weigh_more_and_ascii_terms_get_one_extra() -> None:
    weights = {term.text: term.weight for term in keyword_query("報名日期 ab", 3).terms}

    assert weights == {"ab": 3, "報名日期": 4, "報名": 2, "名日": 2, "日期": 2}
    assert keyword_query("問題", 5).min_score == 5
