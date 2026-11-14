"""E5 編碼器：切塊的 token 計數、段落 embedding 與查詢 embedding 共用同一個模型。

段落加 `passage: ` 前綴，查詢加 `query: ` 前綴。向量經過正規化，只在 CPU 上執行。模型在第一次
使用（或 helpdesk-api 啟動時呼叫 `load`）才載入，之後同一個程序不再重載；CLI 的掃描與查詢指令
因此不需要載入模型。超過 token 上限的輸入一律拒絕，不讓 tokenizer 靜默截斷。
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any, Final, cast

from helpdesk.application.retrieval.ports import QueryTooLongError
from helpdesk.domain.ingestion import InvalidDocumentError

PASSAGE_PREFIX: Final = "passage: "
QUERY_PREFIX: Final = "query: "


class E5Encoder:
    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        max_sequence_length: int,
        batch_size: int,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._max_sequence_length = max_sequence_length
        self._batch_size = batch_size
        self._model: Any | None = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        self._loaded()

    def dimension(self) -> int:
        dimension = self._loaded().get_sentence_embedding_dimension()
        if not isinstance(dimension, int):
            raise RuntimeError("the embedding model did not report a vector dimension")
        return dimension

    def count(self, text: str) -> int:
        """與 `embed` 實際送進模型的 token 數相同，含前綴與特殊 token。"""

        token_ids = self._loaded().tokenizer.encode(
            PASSAGE_PREFIX + text.strip(), add_special_tokens=True, truncation=False
        )
        return len(token_ids)

    def embed(self, passages: Sequence[str]) -> list[list[float]]:
        prepared = [PASSAGE_PREFIX + passage.strip() for passage in passages]
        over_limit = self._over_limit(prepared)
        if over_limit:
            raise InvalidDocumentError(
                f"chunks exceed the embedding token limit at index(es): {over_limit[:10]}"
            )
        return self._encode(prepared)

    def embed_query(self, text: str) -> list[float]:
        prepared = [QUERY_PREFIX + text.strip()]
        if self._over_limit(prepared):
            raise QueryTooLongError(
                f"the question exceeds the {self._max_sequence_length}-token embedding limit"
            )
        return self._encode(prepared)[0]

    def _over_limit(self, prepared: list[str]) -> list[int]:
        tokenized = self._loaded().tokenizer(prepared, add_special_tokens=True, truncation=False)
        return [
            index
            for index, token_ids in enumerate(tokenized["input_ids"])
            if len(token_ids) > self._max_sequence_length
        ]

    def _encode(self, prepared: list[str]) -> list[list[float]]:
        vectors = self._loaded().encode(
            prepared,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return cast(list[list[float]], vectors.tolist())

    def _loaded(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load()
        return self._model

    def _load(self) -> Any:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self._model_id, revision=self._revision, device="cpu")
        model.max_seq_length = self._max_sequence_length
        return model
