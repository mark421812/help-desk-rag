"""從問題取出關鍵字檢索用的詞與權重。

繁體中文沒有空白分詞，也不引入分詞套件。CJK 連續字串取整串（12 字以內）與所有二字組，
英數字（人名、代碼、型號）取整個 token 並轉小寫。詞越長越具體；英數詞多加 1，讓代碼比單一
二字組更有分量。
"""

from __future__ import annotations

import re
from typing import Final

from helpdesk.application.retrieval.ports import KeywordQuery, KeywordTerm

_CJK_RUN: Final = re.compile(r"[㐀-䶿一-鿿豈-﫿]+")
_ASCII_TOKEN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-./]*[A-Za-z0-9]|[A-Za-z0-9]")
_MAX_TERMS: Final = 32
_MAX_WHOLE_RUN: Final = 12


def keyword_query(question: str, min_score: int) -> KeywordQuery:
    return KeywordQuery(
        terms=tuple(KeywordTerm(term, _weight(term)) for term in _terms(question)),
        min_score=min_score,
    )


def _terms(question: str) -> list[str]:
    """去重並保留出現順序，最多 32 個。"""

    terms: dict[str, None] = {}
    for token in _ASCII_TOKEN.findall(question):
        if len(token) >= 2:
            terms[token.lower()] = None
    for run in _CJK_RUN.findall(question):
        if len(run) < 2:
            continue
        if len(run) <= _MAX_WHOLE_RUN:
            terms[run] = None
        for index in range(len(run) - 1):
            terms[run[index : index + 2]] = None
    return list(terms)[:_MAX_TERMS]


def _weight(term: str) -> int:
    return len(term) + (1 if term.isascii() else 0)
