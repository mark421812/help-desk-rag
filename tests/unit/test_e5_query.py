"""E5 查詢 embedding：加 `query:` 前綴、同一個程序只載入一次模型、超長問題被拒絕。

主機上沒有 torch，所以用替身模型；真正的 E5 在 Docker 映像裡驗證。
"""

from __future__ import annotations

from typing import Any

import pytest

from helpdesk.application.retrieval.ports import QueryTooLongError
from helpdesk.infrastructure.embedding.e5 import E5Encoder


class StubVectors:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[float]]:
        return self._rows


class StubModel:
    def __init__(self) -> None:
        self.encoded: list[str] = []

    def tokenizer(self, texts: list[str], **_: Any) -> dict[str, list[list[int]]]:
        return {"input_ids": [[0] * len(text) for text in texts]}

    def encode(self, texts: list[str], **options: Any) -> StubVectors:
        assert options["normalize_embeddings"] is True
        self.encoded.extend(texts)
        return StubVectors([[1.0, 0.0] for _ in texts])


@pytest.fixture
def encoder(monkeypatch: pytest.MonkeyPatch) -> tuple[E5Encoder, StubModel, list[int]]:
    model = StubModel()
    loads: list[int] = []

    def load(self: E5Encoder) -> StubModel:
        loads.append(1)
        return model

    monkeypatch.setattr(E5Encoder, "_load", load)
    return E5Encoder("e5", "rev", max_sequence_length=40, batch_size=8), model, loads


def test_queries_get_the_query_prefix_and_reuse_one_loaded_model(
    encoder: tuple[E5Encoder, StubModel, list[int]],
) -> None:
    e5, model, loads = encoder

    assert not e5.is_loaded
    assert e5.embed_query("  報名時間 ") == [1.0, 0.0]
    e5.embed_query("放榜")
    e5.embed(["段落"])

    assert model.encoded == ["query: 報名時間", "query: 放榜", "passage: 段落"]
    assert loads == [1]
    assert e5.is_loaded


def test_a_question_over_the_token_limit_is_rejected(
    encoder: tuple[E5Encoder, StubModel, list[int]],
) -> None:
    e5, model, _ = encoder

    with pytest.raises(QueryTooLongError):
        e5.embed_query("問" * 40)

    assert model.encoded == []
